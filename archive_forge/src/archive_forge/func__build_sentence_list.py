import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _build_sentence_list(self, text: str, tokens: Iterator[PunktToken]) -> Iterator[str]:
    """
        Given the original text and the list of augmented word tokens,
        construct and return a tokenized list of sentence strings.
        """
    pos = 0
    white_space_regexp = re.compile('\\s*')
    sentence = ''
    for aug_tok in tokens:
        tok = aug_tok.tok
        white_space = white_space_regexp.match(text, pos).group()
        pos += len(white_space)
        if text[pos:pos + len(tok)] != tok:
            pat = '\\s*'.join((re.escape(c) for c in tok))
            m = re.compile(pat).match(text, pos)
            if m:
                tok = m.group()
        assert text[pos:pos + len(tok)] == tok
        pos += len(tok)
        if sentence:
            sentence += white_space
        sentence += tok
        if aug_tok.sentbreak:
            yield sentence
            sentence = ''
    if sentence:
        yield sentence