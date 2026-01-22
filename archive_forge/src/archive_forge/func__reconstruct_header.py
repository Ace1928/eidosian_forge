from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
def _reconstruct_header(cls_name, bases, value):
    return type(cls_name, bases, {})._reconstruct(value)