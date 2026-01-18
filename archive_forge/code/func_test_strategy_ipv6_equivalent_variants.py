import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
@pytest.mark.parametrize('str_value', ('2001:0db8:0000:0000:0000:0000:1428:57ab', '2001:0db8:0000:0000:0000::1428:57ab', '2001:0db8:0:0:0:0:1428:57ab', '2001:0db8:0:0::1428:57ab', '2001:0db8::1428:57ab', '2001:0DB8:0000:0000:0000:0000:1428:57AB', '2001:DB8::1428:57AB'))
def test_strategy_ipv6_equivalent_variants(str_value):
    assert ipv6.str_to_int(str_value) == 42540766411282592856903984951992014763