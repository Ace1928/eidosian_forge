from srsly.msgpack import packb, Unpacker, OutOfData
def test_read_map_header():
    unpacker = Unpacker()
    unpacker.feed(packb({'a': 'A'}))
    assert unpacker.read_map_header() == 1
    assert unpacker.unpack() == b'a'
    assert unpacker.unpack() == b'A'
    try:
        unpacker.unpack()
        assert 0, 'should raise exception'
    except OutOfData:
        assert 1, 'okay'