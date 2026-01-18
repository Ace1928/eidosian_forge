import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def train_tokens(self, tokens, verbose=False, finalize=True):
    """
        Collects training data from a given list of tokens.
        """
    self._train_tokens((self._Token(t) for t in tokens), verbose)
    if finalize:
        self.finalize_training(verbose)