import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class CharsetMetaAttributeValue(AttributeValueWithCharsetSubstitution):
    """A generic stand-in for the value of a meta tag's 'charset' attribute.

    When Beautiful Soup parses the markup '<meta charset="utf8">', the
    value of the 'charset' attribute will be one of these objects.
    """

    def __new__(cls, original_value):
        obj = str.__new__(cls, original_value)
        obj.original_value = original_value
        return obj

    def encode(self, encoding):
        """When an HTML document is being encoded to a given encoding, the
        value of a meta tag's 'charset' is the name of the encoding.
        """
        if encoding in PYTHON_SPECIFIC_ENCODINGS:
            return ''
        return encoding