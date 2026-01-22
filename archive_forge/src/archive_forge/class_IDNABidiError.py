from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
class IDNABidiError(IDNAError):
    """ Exception when bidirectional requirements are not satisfied """
    pass