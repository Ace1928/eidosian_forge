from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
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