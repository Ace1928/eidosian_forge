from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
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