import pytest
from bs4.element import (
from . import SoupTest
def test_ruby_strings(self):
    markup = '<ruby>漢 <rp>(</rp><rt>kan</rt><rp>)</rp> 字 <rp>(</rp><rt>ji</rt><rp>)</rp></ruby>'
    soup = self.soup(markup)
    assert isinstance(soup.rp.string, RubyParenthesisString)
    assert isinstance(soup.rt.string, RubyTextString)
    assert '漢字' == soup.get_text(strip=True)
    assert '漢(kan)字(ji)' == soup.get_text(strip=True, types=(NavigableString, RubyTextString, RubyParenthesisString))