import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_select_duplicate_elements(self):
    markup = '<div class="c1"/><div class="c2"/><div class="c1"/>'
    soup = BeautifulSoup(markup, 'html.parser')
    selected = soup.select('.c1, .c2')
    assert 3 == len(selected)
    for element in soup.find_all(class_=['c1', 'c2']):
        assert element in selected