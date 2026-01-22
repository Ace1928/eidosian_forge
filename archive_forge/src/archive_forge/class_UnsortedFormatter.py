import pytest
from bs4.element import Tag
from bs4.formatter import (
from . import SoupTest
class UnsortedFormatter(Formatter):

    def attributes(self, tag):
        self.called_with = tag
        for k, v in sorted(tag.attrs.items()):
            if k == 'ignore':
                continue
            yield (k, v)