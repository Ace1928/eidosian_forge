import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
def test_tracking_line_numbers(self):
    soup = self.soup('\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>', store_line_numbers=True)
    assert 'sourceline' == soup.p.sourceline.name
    assert 'sourcepos' == soup.p.sourcepos.name