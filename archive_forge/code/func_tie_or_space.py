from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
def tie_or_space(word, tie='~', space=' '):
    if bibtex_len(word) < enough_chars:
        return tie
    else:
        return space