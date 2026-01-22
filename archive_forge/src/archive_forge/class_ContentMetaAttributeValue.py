import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class ContentMetaAttributeValue(AttributeValueWithCharsetSubstitution):
    """A generic stand-in for the value of a meta tag's 'content' attribute.

    When Beautiful Soup parses the markup:
     <meta http-equiv="content-type" content="text/html; charset=utf8">

    The value of the 'content' attribute will be one of these objects.
    """
    CHARSET_RE = re.compile('((^|;)\\s*charset=)([^;]*)', re.M)

    def __new__(cls, original_value):
        match = cls.CHARSET_RE.search(original_value)
        if match is None:
            return str.__new__(str, original_value)
        obj = str.__new__(cls, original_value)
        obj.original_value = original_value
        return obj

    def encode(self, encoding):
        if encoding in PYTHON_SPECIFIC_ENCODINGS:
            return ''

        def rewrite(match):
            return match.group(1) + encoding
        return self.CHARSET_RE.sub(rewrite, self.original_value)