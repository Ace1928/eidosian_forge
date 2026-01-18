import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def test_pairlist():
    pairlist = [(b'a', 1), (2, b'b'), (b'foo', b'bar')]
    packer = Packer()
    packed = packer.pack_map_pairs(pairlist)
    unpacked = unpackb(packed, object_pairs_hook=list)
    assert pairlist == unpacked