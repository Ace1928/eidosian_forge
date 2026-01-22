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
def _assert_no_parser_specified(self, w):
    warning = self._assert_warning(w, GuessedAtParserWarning)
    message = str(warning.message)
    assert message.startswith(BeautifulSoup.NO_PARSER_SPECIFIED_WARNING[:60])