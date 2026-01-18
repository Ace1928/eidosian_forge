import pytest
from netaddr import INET_ATON, INET_PTON, AddrFormatError
from netaddr.strategy import ipv4
def test_strategy_inet_aton_behaviour():
    assert ipv4.str_to_int('127', flags=INET_ATON) == 127
    assert ipv4.str_to_int('0x7f', flags=INET_ATON) == 127
    assert ipv4.str_to_int('0177', flags=INET_ATON) == 127
    assert ipv4.str_to_int('127.1', flags=INET_ATON) == 2130706433
    assert ipv4.str_to_int('0x7f.1', flags=INET_ATON) == 2130706433
    assert ipv4.str_to_int('0177.1', flags=INET_ATON) == 2130706433
    assert ipv4.str_to_int('127.0.0.1', flags=INET_ATON) == 2130706433