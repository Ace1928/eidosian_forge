import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _annotate_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
    """
        Given a set of tokens augmented with markers for line-start and
        paragraph-start, returns an iterator through those tokens with full
        annotation including predicted sentence breaks.
        """
    tokens = self._annotate_first_pass(tokens)
    tokens = self._annotate_second_pass(tokens)
    return tokens