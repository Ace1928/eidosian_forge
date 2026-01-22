import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@property
def _re_non_word_chars(self):
    return '(?:[)\\";}\\]\\*:@\\\'\\({\\[%s])' % re.escape(''.join(set(self.sent_end_chars) - {'.'}))