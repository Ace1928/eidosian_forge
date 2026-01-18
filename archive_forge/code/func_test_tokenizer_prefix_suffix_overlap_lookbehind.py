import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
def test_tokenizer_prefix_suffix_overlap_lookbehind(en_vocab):
    prefixes = ['a(?=.)']
    suffixes = ['(?<=\\w)\\.', '(?<=a)\\d+\\.']
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)
    tokenizer = Tokenizer(en_vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search)
    tokens = [t.text for t in tokenizer('a10.')]
    assert tokens == ['a', '10', '.']
    explain_tokens = [t[1] for t in tokenizer.explain('a10.')]
    assert tokens == explain_tokens