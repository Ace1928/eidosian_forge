from array import array
from srsly.msgpack import packb, unpackb
def test_str32_from_float():
    _runtest('f', 2 ** 16, b'\xdb', b'\x00\x01\x00\x00', False)