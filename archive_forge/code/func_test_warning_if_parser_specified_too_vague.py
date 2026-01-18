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
def test_warning_if_parser_specified_too_vague(self):
    with warnings.catch_warnings(record=True) as w:
        soup = BeautifulSoup('<a><b></b></a>', 'html')
    self._assert_no_parser_specified(w)