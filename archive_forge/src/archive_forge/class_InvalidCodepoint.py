from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
class InvalidCodepoint(IDNAError):
    """ Exception when a disallowed or unallocated codepoint is used """
    pass