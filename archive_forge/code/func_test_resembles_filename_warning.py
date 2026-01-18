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
@pytest.mark.parametrize('extension', ['markup.html', 'markup.htm', 'markup.HTML', 'markup.txt', 'markup.xhtml', 'markup.xml', '/home/user/file', 'c:\\user\x0cile'])
def test_resembles_filename_warning(self, extension):
    with warnings.catch_warnings(record=True) as w:
        soup = BeautifulSoup('markup' + extension, 'html.parser')
        warning = self._assert_warning(w, MarkupResemblesLocatorWarning)
        assert 'looks more like a filename' in str(warning.message)