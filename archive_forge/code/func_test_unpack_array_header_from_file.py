from io import BytesIO
import sys
import pytest
from srsly.msgpack import Unpacker, packb, OutOfData, ExtType
def test_unpack_array_header_from_file():
    f = BytesIO(packb([1, 2, 3, 4]))
    unpacker = Unpacker(f)
    assert unpacker.read_array_header() == 4
    assert unpacker.unpack() == 1
    assert unpacker.unpack() == 2
    assert unpacker.unpack() == 3
    assert unpacker.unpack() == 4
    with pytest.raises(OutOfData):
        unpacker.unpack()