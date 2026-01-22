import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _is_rare_abbrev_type(self, cur_tok, next_tok):
    """
        A word type is counted as a rare abbreviation if...
          - it's not already marked as an abbreviation
          - it occurs fewer than ABBREV_BACKOFF times
          - either it is followed by a sentence-internal punctuation
            mark, *or* it is followed by a lower-case word that
            sometimes appears with upper case, but never occurs with
            lower case at the beginning of sentences.
        """
    if cur_tok.abbr or not cur_tok.sentbreak:
        return False
    typ = cur_tok.type_no_sentperiod
    count = self._type_fdist[typ] + self._type_fdist[typ[:-1]]
    if typ in self._params.abbrev_types or count >= self.ABBREV_BACKOFF:
        return False
    if next_tok.tok[:1] in self._lang_vars.internal_punctuation:
        return True
    if next_tok.first_lower:
        typ2 = next_tok.type_no_sentperiod
        typ2ortho_context = self._params.ortho_context[typ2]
        if typ2ortho_context & _ORTHO_BEG_UC and (not typ2ortho_context & _ORTHO_MID_UC):
            return True