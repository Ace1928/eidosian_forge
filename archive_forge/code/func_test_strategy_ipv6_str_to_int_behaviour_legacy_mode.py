import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
def test_strategy_ipv6_str_to_int_behaviour_legacy_mode():
    assert ipv6.str_to_int('::127') == 295
    with pytest.raises(AddrFormatError):
        ipv6.str_to_int('::0x7f')
    assert ipv6.str_to_int('::0177') == 375
    with pytest.raises(AddrFormatError):
        ipv6.str_to_int('::127.1')
    with pytest.raises(AddrFormatError):
        ipv6.str_to_int('::0x7f.1')
    with pytest.raises(AddrFormatError):
        ipv6.str_to_int('::0177.1')
    assert ipv6.str_to_int('::127.0.0.1') == 2130706433