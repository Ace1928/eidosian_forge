from unittest.mock import patch
from docstring_parser import parse_from_object
class WithoutType:
    attr_one = 'value'
    'Description for attr_one'