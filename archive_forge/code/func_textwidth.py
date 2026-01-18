import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Generator
from unicodedata import east_asian_width
from docutils.parsers.rst import roles
from docutils.parsers.rst.languages import en as english
from docutils.statemachine import StringList
from docutils.utils import Reporter
from jinja2 import Environment
from sphinx.locale import __
from sphinx.util import docutils, logging
def textwidth(text: str, widechars: str='WF') -> int:
    """Get width of text."""

    def charwidth(char: str, widechars: str) -> int:
        if east_asian_width(char) in widechars:
            return 2
        else:
            return 1
    return sum((charwidth(c, widechars) for c in text))