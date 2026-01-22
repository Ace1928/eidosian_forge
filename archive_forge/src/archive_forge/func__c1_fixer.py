import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def _c1_fixer(match):
    return match.group(0).encode('latin-1').decode('sloppy-windows-1252')