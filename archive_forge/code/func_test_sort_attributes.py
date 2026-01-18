import pytest
from bs4.element import Tag
from bs4.formatter import (
from . import SoupTest
def test_sort_attributes(self):

    class UnsortedFormatter(Formatter):

        def attributes(self, tag):
            self.called_with = tag
            for k, v in sorted(tag.attrs.items()):
                if k == 'ignore':
                    continue
                yield (k, v)
    soup = self.soup('<p cval="1" aval="2" ignore="ignored"></p>')
    formatter = UnsortedFormatter()
    decoded = soup.decode(formatter=formatter)
    assert formatter.called_with == soup.p
    assert '<p aval="2" cval="1"></p>' == decoded