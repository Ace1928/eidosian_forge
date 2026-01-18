import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
@pytest.mark.parametrize('str_value', ([], (), {}, True, False, 192))
def test_valid_str_unexpected_types(str_value):
    with pytest.raises(TypeError):
        ipv6.valid_str(str_value)