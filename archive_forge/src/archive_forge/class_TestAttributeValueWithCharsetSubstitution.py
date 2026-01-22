from bs4.element import (
from . import SoupTest
class TestAttributeValueWithCharsetSubstitution(object):
    """Certain attributes are designed to have the charset of the
    final document substituted into their value.
    """

    def test_content_meta_attribute_value(self):
        value = CharsetMetaAttributeValue('euc-jp')
        assert 'euc-jp' == value
        assert 'euc-jp' == value.original_value
        assert 'utf8' == value.encode('utf8')
        assert 'ascii' == value.encode('ascii')

    def test_content_meta_attribute_value(self):
        value = ContentMetaAttributeValue('text/html; charset=euc-jp')
        assert 'text/html; charset=euc-jp' == value
        assert 'text/html; charset=euc-jp' == value.original_value
        assert 'text/html; charset=utf8' == value.encode('utf8')
        assert 'text/html; charset=ascii' == value.encode('ascii')