from srsly.msgpack import packb, unpackb, ExtType
def test_str8():
    header = b'\xd9'
    data = b'x' * 32
    b = packb(data.decode(), use_bin_type=True)
    assert len(b) == len(data) + 2
    assert b[0:2] == header + b' '
    assert b[2:] == data
    assert unpackb(b) == data
    data = b'x' * 255
    b = packb(data.decode(), use_bin_type=True)
    assert len(b) == len(data) + 2
    assert b[0:2] == header + b'\xff'
    assert b[2:] == data
    assert unpackb(b) == data