import re
import warnings
from typing import Iterator, List, Tuple
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import align_tokens
class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """
    CONTRACTIONS2 = ['(?i)\\b(can)(?#X)(not)\\b', "(?i)\\b(d)(?#X)('ye)\\b", '(?i)\\b(gim)(?#X)(me)\\b', '(?i)\\b(gon)(?#X)(na)\\b', '(?i)\\b(got)(?#X)(ta)\\b', '(?i)\\b(lem)(?#X)(me)\\b', "(?i)\\b(more)(?#X)('n)\\b", '(?i)\\b(wan)(?#X)(na)(?=\\s)']
    CONTRACTIONS3 = ["(?i) ('t)(?#X)(is)\\b", "(?i) ('t)(?#X)(was)\\b"]
    CONTRACTIONS4 = ['(?i)\\b(whad)(dd)(ya)\\b', '(?i)\\b(wha)(t)(cha)\\b']