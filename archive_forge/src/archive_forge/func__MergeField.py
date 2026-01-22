import encodings.raw_unicode_escape  # pylint: disable=unused-import
import encodings.unicode_escape  # pylint: disable=unused-import
import io
import math
import re
from google.protobuf.internal import decoder
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import text_encoding
from google.protobuf import unknown_fields
def _MergeField(self, tokenizer, message):
    """Merges a single protocol message field into a message.

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      message: A protocol message to record the data.

    Raises:
      ParseError: In case of text parsing problems.
    """
    message_descriptor = message.DESCRIPTOR
    if message_descriptor.full_name == _ANY_FULL_TYPE_NAME and tokenizer.TryConsume('['):
        type_url_prefix, packed_type_name = self._ConsumeAnyTypeUrl(tokenizer)
        tokenizer.Consume(']')
        tokenizer.TryConsume(':')
        self._DetectSilentMarker(tokenizer, message_descriptor.full_name, type_url_prefix + '/' + packed_type_name)
        if tokenizer.TryConsume('<'):
            expanded_any_end_token = '>'
        else:
            tokenizer.Consume('{')
            expanded_any_end_token = '}'
        expanded_any_sub_message = _BuildMessageFromTypeName(packed_type_name, self.descriptor_pool)
        if expanded_any_sub_message is None:
            raise ParseError('Type %s not found in descriptor pool' % packed_type_name)
        while not tokenizer.TryConsume(expanded_any_end_token):
            if tokenizer.AtEnd():
                raise tokenizer.ParseErrorPreviousToken('Expected "%s".' % (expanded_any_end_token,))
            self._MergeField(tokenizer, expanded_any_sub_message)
        deterministic = False
        message.Pack(expanded_any_sub_message, type_url_prefix=type_url_prefix, deterministic=deterministic)
        return
    if tokenizer.TryConsume('['):
        name = [tokenizer.ConsumeIdentifier()]
        while tokenizer.TryConsume('.'):
            name.append(tokenizer.ConsumeIdentifier())
        name = '.'.join(name)
        if not message_descriptor.is_extendable:
            raise tokenizer.ParseErrorPreviousToken('Message type "%s" does not have extensions.' % message_descriptor.full_name)
        field = message.Extensions._FindExtensionByName(name)
        if not field:
            if self.allow_unknown_extension:
                field = None
            else:
                raise tokenizer.ParseErrorPreviousToken('Extension "%s" not registered. Did you import the _pb2 module which defines it? If you are trying to place the extension in the MessageSet field of another message that is in an Any or MessageSet field, that message\'s _pb2 module must be imported as well' % name)
        elif message_descriptor != field.containing_type:
            raise tokenizer.ParseErrorPreviousToken('Extension "%s" does not extend message type "%s".' % (name, message_descriptor.full_name))
        tokenizer.Consume(']')
    else:
        name = tokenizer.ConsumeIdentifierOrNumber()
        if self.allow_field_number and name.isdigit():
            number = ParseInteger(name, True, True)
            field = message_descriptor.fields_by_number.get(number, None)
            if not field and message_descriptor.is_extendable:
                field = message.Extensions._FindExtensionByNumber(number)
        else:
            field = message_descriptor.fields_by_name.get(name, None)
            if not field:
                field = message_descriptor.fields_by_name.get(name.lower(), None)
                if field and field.type != descriptor.FieldDescriptor.TYPE_GROUP:
                    field = None
            if field and field.type == descriptor.FieldDescriptor.TYPE_GROUP and (field.message_type.name != name):
                field = None
        if not field and (not self.allow_unknown_field):
            raise tokenizer.ParseErrorPreviousToken('Message type "%s" has no field named "%s".' % (message_descriptor.full_name, name))
    if field:
        if not self._allow_multiple_scalars and field.containing_oneof:
            which_oneof = message.WhichOneof(field.containing_oneof.name)
            if which_oneof is not None and which_oneof != field.name:
                raise tokenizer.ParseErrorPreviousToken('Field "%s" is specified along with field "%s", another member of oneof "%s" for message type "%s".' % (field.name, which_oneof, field.containing_oneof.name, message_descriptor.full_name))
        if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            tokenizer.TryConsume(':')
            self._DetectSilentMarker(tokenizer, message_descriptor.full_name, field.full_name)
            merger = self._MergeMessageField
        else:
            tokenizer.Consume(':')
            self._DetectSilentMarker(tokenizer, message_descriptor.full_name, field.full_name)
            merger = self._MergeScalarField
        if field.label == descriptor.FieldDescriptor.LABEL_REPEATED and tokenizer.TryConsume('['):
            if not tokenizer.TryConsume(']'):
                while True:
                    merger(tokenizer, message, field)
                    if tokenizer.TryConsume(']'):
                        break
                    tokenizer.Consume(',')
        else:
            merger(tokenizer, message, field)
    else:
        assert self.allow_unknown_extension or self.allow_unknown_field
        self._SkipFieldContents(tokenizer, name, message_descriptor.full_name)
    if not tokenizer.TryConsume(','):
        tokenizer.TryConsume(';')