import pytest
from netaddr import INET_ATON, INET_PTON, AddrFormatError
from netaddr.strategy import ipv4
def test_strategy_ipv4():
    b = '11000000.00000000.00000010.00000001'
    i = 3221225985
    t = (192, 0, 2, 1)
    s = '192.0.2.1'
    bin_val = '0b11000000000000000000001000000001'
    p = b'\xc0\x00\x02\x01'
    assert ipv4.bits_to_int(b) == i
    assert ipv4.int_to_bits(i) == b
    assert ipv4.int_to_str(i) == s
    assert ipv4.int_to_words(i) == t
    assert ipv4.int_to_bin(i) == bin_val
    assert ipv4.int_to_bin(i) == bin_val
    assert ipv4.bin_to_int(bin_val) == i
    assert ipv4.words_to_int(t) == i
    assert ipv4.words_to_int(list(t)) == i
    assert ipv4.valid_bin(bin_val)
    assert ipv4.int_to_packed(i) == p
    assert ipv4.packed_to_int(p) == i