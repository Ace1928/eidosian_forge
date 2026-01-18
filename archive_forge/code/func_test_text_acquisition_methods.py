import pytest
from bs4.element import (
from . import SoupTest
def test_text_acquisition_methods(self):
    s = NavigableString('fee ')
    cdata = CData('fie ')
    comment = Comment('foe ')
    assert 'fee ' == s.get_text()
    assert 'fee' == s.get_text(strip=True)
    assert ['fee '] == list(s.strings)
    assert ['fee'] == list(s.stripped_strings)
    assert ['fee '] == list(s._all_strings())
    assert 'fie ' == cdata.get_text()
    assert 'fie' == cdata.get_text(strip=True)
    assert ['fie '] == list(cdata.strings)
    assert ['fie'] == list(cdata.stripped_strings)
    assert ['fie '] == list(cdata._all_strings())
    assert '' == comment.get_text()
    assert [] == list(comment.strings)
    assert [] == list(comment.stripped_strings)
    assert [] == list(comment._all_strings())
    assert 'foe' == comment.get_text(strip=True, types=Comment)
    assert 'foe ' == comment.get_text(types=(Comment, NavigableString))