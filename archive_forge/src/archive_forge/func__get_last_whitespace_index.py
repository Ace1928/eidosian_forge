import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _get_last_whitespace_index(self, text: str) -> int:
    """
        Given a text, find the index of the *last* occurrence of *any*
        whitespace character, i.e. " ", "
", "	", "\r", etc.
        If none is found, return 0.
        """
    for i in range(len(text) - 1, -1, -1):
        if text[i] in string.whitespace:
            return i
    return 0