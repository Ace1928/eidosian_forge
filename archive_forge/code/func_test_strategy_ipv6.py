import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
def test_strategy_ipv6():
    b = '0000000000000000:0000000000000000:0000000000000000:0000000000000000:0000000000000000:0000000000000000:1111111111111111:1111111111111110'
    i = 4294967294
    t = (0, 0, 0, 0, 0, 0, 65535, 65534)
    s = '::255.255.255.254'
    p = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xfe'
    assert ipv6.bits_to_int(b) == i
    assert ipv6.int_to_bits(i) == b
    assert ipv6.int_to_str(i) == s
    assert ipv6.str_to_int(s) == i
    assert ipv6.int_to_words(i) == t
    assert ipv6.words_to_int(t) == i
    assert ipv6.words_to_int(list(t)) == i
    assert ipv6.int_to_packed(i) == p
    assert ipv6.packed_to_int(p) == 4294967294