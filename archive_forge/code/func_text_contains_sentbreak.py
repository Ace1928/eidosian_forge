import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def text_contains_sentbreak(self, text: str) -> bool:
    """
        Returns True if the given text includes a sentence break.
        """
    found = False
    for tok in self._annotate_tokens(self._tokenize_words(text)):
        if found:
            return True
        if tok.sentbreak:
            found = True
    return False