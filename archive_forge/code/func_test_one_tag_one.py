import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_one_tag_one(self):
    els = self.soup.select('title')
    assert len(els) == 1
    assert els[0].name == 'title'
    assert els[0].contents == ['The title']