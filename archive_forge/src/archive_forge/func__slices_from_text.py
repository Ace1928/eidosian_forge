import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _slices_from_text(self, text: str) -> Iterator[slice]:
    last_break = 0
    for match, context in self._match_potential_end_contexts(text):
        if self.text_contains_sentbreak(context):
            yield slice(last_break, match.end())
            if match.group('next_tok'):
                last_break = match.start('next_tok')
            else:
                last_break = match.end()
    yield slice(last_break, len(text.rstrip()))