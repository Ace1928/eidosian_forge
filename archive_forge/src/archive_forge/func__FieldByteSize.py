from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
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