from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
def test_url_warning_with_unicode_and_space(self):
    with warnings.catch_warnings(record=True) as warning_list:
        soup = self.soup('http://www.crummyunicode.com/ is great')
    assert not any(('looks more like a URL' in str(w.message) for w in warning_list))