from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def _bibtex_abbreviate():
    for token in split_tex_string(string, sep=separator):
        letter = bibtex_first_letter(token)
        if letter:
            yield letter