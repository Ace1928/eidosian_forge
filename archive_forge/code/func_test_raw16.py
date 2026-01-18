from srsly.msgpack import packb, unpackb
def test_raw16():
    check_raw(3, 1 << 5)
    check_raw(3, (1 << 16) - 1)