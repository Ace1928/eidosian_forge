import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_normal_doctypes(self):
    """Make sure normal, everyday HTML doctypes are handled correctly."""
    self.assertDoctypeHandled('html')
    self.assertDoctypeHandled('html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"')