import re
from threading import Lock
from io import TextIOBase
from sqlparse import tokens, keywords
from sqlparse.utils import consume
def add_keywords(self, keywords):
    """Add keyword dictionaries. Keywords are looked up in the same order
        that dictionaries were added."""
    self._keywords.append(keywords)