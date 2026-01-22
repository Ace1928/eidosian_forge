from bs4.element import (
from . import SoupTest
class TestNamedspacedAttribute(object):

    def test_name_may_be_none_or_missing(self):
        a = NamespacedAttribute('xmlns', None)
        assert a == 'xmlns'
        a = NamespacedAttribute('xmlns', '')
        assert a == 'xmlns'
        a = NamespacedAttribute('xmlns')
        assert a == 'xmlns'

    def test_namespace_may_be_none_or_missing(self):
        a = NamespacedAttribute(None, 'tag')
        assert a == 'tag'
        a = NamespacedAttribute('', 'tag')
        assert a == 'tag'

    def test_attribute_is_equivalent_to_colon_separated_string(self):
        a = NamespacedAttribute('a', 'b')
        assert 'a:b' == a

    def test_attributes_are_equivalent_if_prefix_and_name_identical(self):
        a = NamespacedAttribute('a', 'b', 'c')
        b = NamespacedAttribute('a', 'b', 'c')
        assert a == b
        c = NamespacedAttribute('a', 'b', None)
        assert a == c
        d = NamespacedAttribute('a', 'z', 'c')
        assert a != d
        e = NamespacedAttribute('z', 'b', 'c')
        assert a != e