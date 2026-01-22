import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class TemplateString(NavigableString):
    """A NavigableString representing a string found inside an HTML
    template embedded in a larger document.

    Used to distinguish such strings from the main body of the document.
    """
    pass