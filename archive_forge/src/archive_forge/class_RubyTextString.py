import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class RubyTextString(NavigableString):
    """A NavigableString representing the contents of the <rt> HTML
    element.

    https://dev.w3.org/html5/spec-LC/text-level-semantics.html#the-rt-element

    Can be used to distinguish such strings from the strings they're
    annotating.
    """
    pass