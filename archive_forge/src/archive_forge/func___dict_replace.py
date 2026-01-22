import os, urllib.parse, urllib.request
import io
import codecs
from . import handler
from . import xmlreader
def __dict_replace(s, d):
    """Replace substrings of a string using a dictionary."""
    for key, value in d.items():
        s = s.replace(key, value)
    return s