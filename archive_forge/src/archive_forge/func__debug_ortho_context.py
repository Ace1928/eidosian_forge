import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _debug_ortho_context(self, typ):
    context = self.ortho_context[typ]
    if context & _ORTHO_BEG_UC:
        yield 'BEG-UC'
    if context & _ORTHO_MID_UC:
        yield 'MID-UC'
    if context & _ORTHO_UNK_UC:
        yield 'UNK-UC'
    if context & _ORTHO_BEG_LC:
        yield 'BEG-LC'
    if context & _ORTHO_MID_LC:
        yield 'MID-LC'
    if context & _ORTHO_UNK_LC:
        yield 'UNK-LC'