from __future__ import unicode_literals
import re
from string import ascii_letters, digits
import six
from pybtex import textutils
from pybtex.bibtex.utils import split_name_list
from pybtex.database import Entry, Person, BibliographyDataError
from pybtex.database.input import BaseParser
from pybtex.scanner import (
from pybtex.utils import CaseInsensitiveDict, CaseInsensitiveSet
class BibTeXEntryIterator(LowLevelParser):

    def __init__(self, *args, **kwargs):
        import warnings
        message = 'BibTeXEntryIterator is deprecated since 0.22: renamed to LowLevelParser'
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        super(BibTeXEntryIterator, self).__init__(*args, **kwargs)