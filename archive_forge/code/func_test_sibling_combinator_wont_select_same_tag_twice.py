import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_sibling_combinator_wont_select_same_tag_twice(self):
    self.assert_selects('p[lang] ~ p', ['lang-en-gb', 'lang-en-us', 'lang-fr'])