import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_select_dashed_tag_ids(self):
    self.assert_selects('custom-dashed-tag', ['dash1', 'dash2'])