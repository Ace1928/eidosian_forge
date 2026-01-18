from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
@classmethod
def warn_if_markup_looks_like_xml(cls, markup, stacklevel=3):
    """Perform a check on some markup to see if it looks like XML
        that's not XHTML. If so, issue a warning.

        This is much less reliable than doing the check while parsing,
        but some of the tree builders can't do that.

        :param stacklevel: The stacklevel of the code calling this
        function.

        :return: True if the markup looks like non-XHTML XML, False
        otherwise.

        """
    if isinstance(markup, bytes):
        prefix = cls.XML_PREFIX_B
        looks_like_html = cls.LOOKS_LIKE_HTML_B
    else:
        prefix = cls.XML_PREFIX
        looks_like_html = cls.LOOKS_LIKE_HTML
    if markup is not None and markup.startswith(prefix) and (not looks_like_html.search(markup[:500])):
        cls._warn(stacklevel=stacklevel + 2)
        return True
    return False