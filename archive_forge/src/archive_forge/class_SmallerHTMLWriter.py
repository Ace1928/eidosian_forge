import keyword
import tokenize
from html import escape
from typing import List
from . import reflect
class SmallerHTMLWriter(HTMLWriter):
    """
    HTMLWriter that doesn't generate spans for some junk.

    Results in much smaller HTML output.
    """
    noSpan = ['endmarker', 'indent', 'dedent', 'op', 'newline', 'nl']