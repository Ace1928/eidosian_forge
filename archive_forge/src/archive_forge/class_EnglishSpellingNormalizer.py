import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self, english_spelling_mapping):
        self.mapping = english_spelling_mapping

    def __call__(self, s: str):
        return ' '.join((self.mapping.get(word, word) for word in s.split()))