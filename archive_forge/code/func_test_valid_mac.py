from netaddr import valid_mac, valid_eui64
def test_valid_mac():
    assert valid_mac('00-B0-D0-86-BB-F7')
    assert not valid_mac('00-1B-77-49-54-FD-12-34')