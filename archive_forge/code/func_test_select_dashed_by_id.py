import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_select_dashed_by_id(self):
    dashed = self.soup.select('custom-dashed-tag[id="dash2"]')
    assert dashed[0].name == 'custom-dashed-tag'
    assert dashed[0]['id'] == 'dash2'