import pytest
from bs4.element import (
from . import SoupTest
class TestNavigableStringSubclasses(SoupTest):

    def test_cdata(self):
        soup = self.soup('')
        cdata = CData('foo')
        soup.insert(1, cdata)
        assert str(soup) == '<![CDATA[foo]]>'
        assert soup.find(string='foo') == 'foo'
        assert soup.contents[0] == 'foo'

    def test_cdata_is_never_formatted(self):
        """Text inside a CData object is passed into the formatter.

        But the return value is ignored.
        """
        self.count = 0

        def increment(*args):
            self.count += 1
            return 'BITTER FAILURE'
        soup = self.soup('')
        cdata = CData('<><><>')
        soup.insert(1, cdata)
        assert b'<![CDATA[<><><>]]>' == soup.encode(formatter=increment)
        assert 1 == self.count

    def test_doctype_ends_in_newline(self):
        doctype = Doctype('foo')
        soup = self.soup('')
        soup.insert(1, doctype)
        assert soup.encode() == b'<!DOCTYPE foo>\n'

    def test_declaration(self):
        d = Declaration('foo')
        assert '<?foo?>' == d.output_ready()

    def test_default_string_containers(self):
        soup = self.soup('<div>text</div><script>text</script><style>text</style>')
        assert [NavigableString, Script, Stylesheet] == [x.__class__ for x in soup.find_all(string=True)]
        soup = self.soup('<template>Some text<p>In a tag</p></template>Some text outside')
        assert all((isinstance(x, TemplateString) for x in soup.template._all_strings(types=None)))
        outside = soup.template.next_sibling
        assert isinstance(outside, NavigableString)
        assert not isinstance(outside, TemplateString)
        markup = b'<template>Some text<p>In a tag</p><!--with a comment--></template>'
        soup = self.soup(markup)
        assert markup == soup.template.encode('utf8')

    def test_ruby_strings(self):
        markup = '<ruby>漢 <rp>(</rp><rt>kan</rt><rp>)</rp> 字 <rp>(</rp><rt>ji</rt><rp>)</rp></ruby>'
        soup = self.soup(markup)
        assert isinstance(soup.rp.string, RubyParenthesisString)
        assert isinstance(soup.rt.string, RubyTextString)
        assert '漢字' == soup.get_text(strip=True)
        assert '漢(kan)字(ji)' == soup.get_text(strip=True, types=(NavigableString, RubyTextString, RubyParenthesisString))