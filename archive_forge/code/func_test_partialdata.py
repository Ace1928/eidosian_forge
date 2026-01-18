import io
import pytest
from srsly.msgpack import Unpacker, BufferFull
from srsly.msgpack import pack
from srsly.msgpack.exceptions import OutOfData
def test_partialdata():
    unpacker = Unpacker()
    unpacker.feed(b'\xa5')
    with pytest.raises(StopIteration):
        next(iter(unpacker))
    unpacker.feed(b'h')
    with pytest.raises(StopIteration):
        next(iter(unpacker))
    unpacker.feed(b'a')
    with pytest.raises(StopIteration):
        next(iter(unpacker))
    unpacker.feed(b'l')
    with pytest.raises(StopIteration):
        next(iter(unpacker))
    unpacker.feed(b'l')
    with pytest.raises(StopIteration):
        next(iter(unpacker))
    unpacker.feed(b'o')
    assert next(iter(unpacker)) == b'hallo'