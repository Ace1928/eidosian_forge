from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
@staticmethod
def value_parser(value):
    address_list, value = parser.get_address_list(value)
    assert not value, 'this should not happen'
    return address_list