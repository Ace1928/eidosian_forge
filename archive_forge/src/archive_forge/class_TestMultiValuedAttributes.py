import warnings
from bs4.element import (
from . import SoupTest
class TestMultiValuedAttributes(SoupTest):
    """Test the behavior of multi-valued attributes like 'class'.

    The values of such attributes are always presented as lists.
    """

    def test_single_value_becomes_list(self):
        soup = self.soup("<a class='foo'>")
        assert ['foo'] == soup.a['class']

    def test_multiple_values_becomes_list(self):
        soup = self.soup("<a class='foo bar'>")
        assert ['foo', 'bar'] == soup.a['class']

    def test_multiple_values_separated_by_weird_whitespace(self):
        soup = self.soup("<a class='foo\tbar\nbaz'>")
        assert ['foo', 'bar', 'baz'] == soup.a['class']

    def test_attributes_joined_into_string_on_output(self):
        soup = self.soup("<a class='foo\tbar'>")
        assert b'<a class="foo bar"></a>' == soup.a.encode()

    def test_get_attribute_list(self):
        soup = self.soup("<a id='abc def'>")
        assert ['abc def'] == soup.a.get_attribute_list('id')

    def test_accept_charset(self):
        soup = self.soup('<form accept-charset="ISO-8859-1 UTF-8">')
        assert ['ISO-8859-1', 'UTF-8'] == soup.form['accept-charset']

    def test_cdata_attribute_applying_only_to_one_tag(self):
        data = '<a accept-charset="ISO-8859-1 UTF-8"></a>'
        soup = self.soup(data)
        assert 'ISO-8859-1 UTF-8' == soup.a['accept-charset']

    def test_customization(self):
        soup = self.soup('<a class="foo" id="bar">', multi_valued_attributes={'*': 'id'})
        assert soup.a['class'] == 'foo'
        assert soup.a['id'] == ['bar']

    def test_hidden_tag_is_invisible(self):
        soup = self.soup('<div id="1"><span id="2">a string</span></div>')
        soup.span.hidden = True
        assert '<div id="1">a string</div>' == str(soup.div)