import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def StyleSanitizer():
    safe_attrs = HTMLSanitizer.SAFE_ATTRS | frozenset(['style'])
    return HTMLSanitizer(safe_attrs=safe_attrs)