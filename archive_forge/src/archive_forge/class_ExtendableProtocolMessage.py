from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
class ExtendableProtocolMessage(ProtocolMessage):

    def HasExtension(self, extension):
        """Checks if the message contains a certain non-repeated extension."""
        self._VerifyExtensionIdentifier(extension)
        return extension in self._extension_fields

    def ClearExtension(self, extension):
        """Clears the value of extension, so that HasExtension() returns false or
    ExtensionSize() returns 0."""
        self._VerifyExtensionIdentifier(extension)
        if extension in self._extension_fields:
            del self._extension_fields[extension]

    def GetExtension(self, extension, index=None):
        """Gets the extension value for a certain extension.

    Args:
      extension: The ExtensionIdentifier for the extension.
      index: The index of element to get in a repeated field. Only needed if
          the extension is repeated.

    Returns:
      The value of the extension if exists, otherwise the default value of the
      extension will be returned.
    """
        self._VerifyExtensionIdentifier(extension)
        if extension in self._extension_fields:
            result = self._extension_fields[extension]
        elif extension.is_repeated:
            result = []
        elif extension.composite_cls:
            result = extension.composite_cls()
        else:
            result = extension.default
        if extension.is_repeated:
            result = result[index]
        return result

    def SetExtension(self, extension, *args):
        """Sets the extension value for a certain scalar type extension.

    Arg varies according to extension type:
    - Singular:
      message.SetExtension(extension, value)
    - Repeated:
      message.SetExtension(extension, index, value)
    where
      extension: The ExtensionIdentifier for the extension.
      index: The index of element to set in a repeated field. Only needed if
          the extension is repeated.
      value: The value to set.

    Raises:
      TypeError if a message type extension is given.
    """
        self._VerifyExtensionIdentifier(extension)
        if extension.composite_cls:
            raise TypeError('Cannot assign to extension "%s" because it is a composite type.' % extension.full_name)
        if extension.is_repeated:
            try:
                index, value = args
            except ValueError:
                raise TypeError('SetExtension(extension, index, value) for repeated extension takes exactly 4 arguments: (%d given)' % (len(args) + 2))
            self._extension_fields[extension][index] = value
        else:
            try:
                value, = args
            except ValueError:
                raise TypeError('SetExtension(extension, value) for singular extension takes exactly 3 arguments: (%d given)' % (len(args) + 2))
            self._extension_fields[extension] = value

    def MutableExtension(self, extension, index=None):
        """Gets a mutable reference of a message type extension.

    For repeated extension, index must be specified, and only one element will
    be returned. For optional extension, if the extension does not exist, a new
    message will be created and set in parent message.

    Args:
      extension: The ExtensionIdentifier for the extension.
      index: The index of element to mutate in a repeated field. Only needed if
          the extension is repeated.

    Returns:
      The mutable message reference.

    Raises:
      TypeError if non-message type extension is given.
    """
        self._VerifyExtensionIdentifier(extension)
        if extension.composite_cls is None:
            raise TypeError('MutableExtension() cannot be applied to "%s", because it is not a composite type.' % extension.full_name)
        if extension.is_repeated:
            if index is None:
                raise TypeError('MutableExtension(extension, index) for repeated extension takes exactly 2 arguments: (1 given)')
            return self.GetExtension(extension, index)
        if extension in self._extension_fields:
            return self._extension_fields[extension]
        else:
            result = extension.composite_cls()
            self._extension_fields[extension] = result
            return result

    def ExtensionList(self, extension):
        """Returns a mutable list of extensions.

    Raises:
      TypeError if the extension is not repeated.
    """
        self._VerifyExtensionIdentifier(extension)
        if not extension.is_repeated:
            raise TypeError('ExtensionList() cannot be applied to "%s", because it is not a repeated extension.' % extension.full_name)
        if extension in self._extension_fields:
            return self._extension_fields[extension]
        result = []
        self._extension_fields[extension] = result
        return result

    def ExtensionSize(self, extension):
        """Returns the size of a repeated extension.

    Raises:
      TypeError if the extension is not repeated.
    """
        self._VerifyExtensionIdentifier(extension)
        if not extension.is_repeated:
            raise TypeError('ExtensionSize() cannot be applied to "%s", because it is not a repeated extension.' % extension.full_name)
        if extension in self._extension_fields:
            return len(self._extension_fields[extension])
        return 0

    def AddExtension(self, extension, value=None):
        """Appends a new element into a repeated extension.

    Arg varies according to the extension field type:
    - Scalar/String:
      message.AddExtension(extension, value)
    - Message:
      mutable_message = AddExtension(extension)

    Args:
      extension: The ExtensionIdentifier for the extension.
      value: The value of the extension if the extension is scalar/string type.
          The value must NOT be set for message type extensions; set values on
          the returned message object instead.

    Returns:
      A mutable new message if it's a message type extension, or None otherwise.

    Raises:
      TypeError if the extension is not repeated, or value is given for message
      type extensions.
    """
        self._VerifyExtensionIdentifier(extension)
        if not extension.is_repeated:
            raise TypeError('AddExtension() cannot be applied to "%s", because it is not a repeated extension.' % extension.full_name)
        if extension in self._extension_fields:
            field = self._extension_fields[extension]
        else:
            field = []
            self._extension_fields[extension] = field
        if extension.composite_cls:
            if value is not None:
                raise TypeError('value must not be set in AddExtension() for "%s", because it is a message type extension. Set values on the returned message instead.' % extension.full_name)
            msg = extension.composite_cls()
            field.append(msg)
            return msg
        field.append(value)

    def _VerifyExtensionIdentifier(self, extension):
        if extension.containing_cls != self.__class__:
            raise TypeError('Containing type of %s is %s, but not %s.' % (extension.full_name, extension.containing_cls.__name__, self.__class__.__name__))

    def _MergeExtensionFields(self, x):
        for ext, val in x._extension_fields.items():
            if ext.is_repeated:
                for single_val in val:
                    if ext.composite_cls is None:
                        self.AddExtension(ext, single_val)
                    else:
                        self.AddExtension(ext).MergeFrom(single_val)
            elif ext.composite_cls is None:
                self.SetExtension(ext, val)
            else:
                self.MutableExtension(ext).MergeFrom(val)

    def _ListExtensions(self):
        return sorted((ext for ext in self._extension_fields if not ext.is_repeated or self.ExtensionSize(ext) > 0), key=lambda item: item.number)

    def _ExtensionEquals(self, x):
        extensions = self._ListExtensions()
        if extensions != x._ListExtensions():
            return False
        for ext in extensions:
            if ext.is_repeated:
                if self.ExtensionSize(ext) != x.ExtensionSize(ext):
                    return False
                for e1, e2 in zip(self.ExtensionList(ext), x.ExtensionList(ext)):
                    if e1 != e2:
                        return False
            elif self.GetExtension(ext) != x.GetExtension(ext):
                return False
        return True

    def _OutputExtensionFields(self, out, partial, extensions, start_index, end_field_number):
        """Serialize a range of extensions.

    To generate canonical output when encoding, we interleave fields and
    extensions to preserve tag order.

    Generated code will prepare a list of ExtensionIdentifier sorted in field
    number order and call this method to serialize a specific range of
    extensions. The range is specified by the two arguments, start_index and
    end_field_number.

    The method will serialize all extensions[i] with i >= start_index and
    extensions[i].number < end_field_number. Since extensions argument is sorted
    by field_number, this is a contiguous range; the first index j not included
    in that range is returned. The return value can be used as the start_index
    in the next call to serialize the next range of extensions.

    Args:
      extensions: A list of ExtensionIdentifier sorted in field number order.
      start_index: The start index in the extensions list.
      end_field_number: The end field number of the extension range.

    Returns:
      The first index that is not in the range. Or the size of extensions if all
      the extensions are within the range.
    """

        def OutputSingleField(ext, value):
            out.putVarInt32(ext.wire_tag)
            if ext.field_type == TYPE_GROUP:
                if partial:
                    value.OutputPartial(out)
                else:
                    value.OutputUnchecked(out)
                out.putVarInt32(ext.wire_tag + 1)
            elif ext.field_type == TYPE_FOREIGN:
                if partial:
                    out.putVarInt32(value.ByteSizePartial())
                    value.OutputPartial(out)
                else:
                    out.putVarInt32(value.ByteSize())
                    value.OutputUnchecked(out)
            else:
                Encoder._TYPE_TO_METHOD[ext.field_type](out, value)
        for ext_index, ext in enumerate(itertools.islice(extensions, start_index, None), start=start_index):
            if ext.number >= end_field_number:
                return ext_index
            if ext.is_repeated:
                for field in self._extension_fields[ext]:
                    OutputSingleField(ext, field)
            else:
                OutputSingleField(ext, self._extension_fields[ext])
        return len(extensions)

    def _ParseOneExtensionField(self, wire_tag, d):
        number = wire_tag >> 3
        if number in self._extensions_by_field_number:
            ext = self._extensions_by_field_number[number]
            if wire_tag != ext.wire_tag:
                return
            if ext.field_type == TYPE_FOREIGN:
                length = d.getVarInt32()
                tmp = Decoder(d.buffer(), d.pos(), d.pos() + length)
                if ext.is_repeated:
                    self.AddExtension(ext).TryMerge(tmp)
                else:
                    self.MutableExtension(ext).TryMerge(tmp)
                d.skip(length)
            elif ext.field_type == TYPE_GROUP:
                if ext.is_repeated:
                    self.AddExtension(ext).TryMerge(d)
                else:
                    self.MutableExtension(ext).TryMerge(d)
            else:
                value = Decoder._TYPE_TO_METHOD[ext.field_type](d)
                if ext.is_repeated:
                    self.AddExtension(ext, value)
                else:
                    self.SetExtension(ext, value)
        else:
            d.skipData(wire_tag)

    def _ExtensionByteSize(self, partial):
        size = 0
        for extension, value in six.iteritems(self._extension_fields):
            ftype = extension.field_type
            tag_size = self.lengthVarInt64(extension.wire_tag)
            if ftype == TYPE_GROUP:
                tag_size *= 2
            if extension.is_repeated:
                size += tag_size * len(value)
                for single_value in value:
                    size += self._FieldByteSize(ftype, single_value, partial)
            else:
                size += tag_size + self._FieldByteSize(ftype, value, partial)
        return size

    def _FieldByteSize(self, ftype, value, partial):
        size = 0
        if ftype == TYPE_STRING:
            size = self.lengthString(len(value))
        elif ftype == TYPE_FOREIGN or ftype == TYPE_GROUP:
            if partial:
                size = self.lengthString(value.ByteSizePartial())
            else:
                size = self.lengthString(value.ByteSize())
        elif ftype == TYPE_INT64 or ftype == TYPE_UINT64 or ftype == TYPE_INT32:
            size = self.lengthVarInt64(value)
        elif ftype in Encoder._TYPE_TO_BYTE_SIZE:
            size = Encoder._TYPE_TO_BYTE_SIZE[ftype]
        else:
            raise AssertionError('Extension type %d is not recognized.' % ftype)
        return size

    def _ExtensionDebugString(self, prefix, printElemNumber):
        res = ''
        extensions = self._ListExtensions()
        for extension in extensions:
            value = self._extension_fields[extension]
            if extension.is_repeated:
                cnt = 0
                for e in value:
                    elm = ''
                    if printElemNumber:
                        elm = '(%d)' % cnt
                    if extension.composite_cls is not None:
                        res += prefix + '[%s%s] {\n' % (extension.full_name, elm)
                        res += e.__str__(prefix + '  ', printElemNumber)
                        res += prefix + '}\n'
            elif extension.composite_cls is not None:
                res += prefix + '[%s] {\n' % extension.full_name
                res += value.__str__(prefix + '  ', printElemNumber)
                res += prefix + '}\n'
            else:
                if extension.field_type in _TYPE_TO_DEBUG_STRING:
                    text_value = _TYPE_TO_DEBUG_STRING[extension.field_type](self, value)
                else:
                    text_value = self.DebugFormat(value)
                res += prefix + '[%s]: %s\n' % (extension.full_name, text_value)
        return res

    @staticmethod
    def _RegisterExtension(cls, extension, composite_cls=None):
        extension.containing_cls = cls
        extension.composite_cls = composite_cls
        if composite_cls is not None:
            extension.message_name = composite_cls._PROTO_DESCRIPTOR_NAME
        actual_handle = cls._extensions_by_field_number.setdefault(extension.number, extension)
        if actual_handle is not extension:
            raise AssertionError('Extensions "%s" and "%s" both try to extend message type "%s" with field number %d.' % (extension.full_name, actual_handle.full_name, cls.__name__, extension.number))