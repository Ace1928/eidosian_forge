from srsly.msgpack import packb, Unpacker, OutOfData
def test_read_array_header():
    unpacker = Unpacker()
    unpacker.feed(packb(['a', 'b', 'c']))
    assert unpacker.read_array_header() == 3
    assert unpacker.unpack() == b'a'
    assert unpacker.unpack() == b'b'
    assert unpacker.unpack() == b'c'
    try:
        unpacker.unpack()
        assert 0, 'should raise exception'
    except OutOfData:
        assert 1, 'okay'