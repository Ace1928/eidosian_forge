import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_precompiled(self):
    sel = self.soup.css.compile('div')
    els = self.soup.select(sel)
    assert len(els) == 4
    for div in els:
        assert div.name == 'div'
    el = self.soup.select_one(sel)
    assert 'main' == el['id']