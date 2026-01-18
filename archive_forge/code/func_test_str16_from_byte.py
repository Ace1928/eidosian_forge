from array import array
from srsly.msgpack import packb, unpackb
def test_str16_from_byte():
    _runtest('B', 2 ** 8, b'\xda', b'\x01\x00', False)
    _runtest('B', 2 ** 16 - 1, b'\xda', b'\xff\xff', False)