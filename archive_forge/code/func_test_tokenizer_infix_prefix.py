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
def test_tokenizer_infix_prefix(en_vocab):
    infixes = ['±']
    suffixes = ['%']
    infix_re = compile_infix_regex(infixes)
    suffix_re = compile_suffix_regex(suffixes)
    tokenizer = Tokenizer(en_vocab, infix_finditer=infix_re.finditer, suffix_search=suffix_re.search)
    tokens = [t.text for t in tokenizer('±10%')]
    assert tokens == ['±10', '%']
    explain_tokens = [t[1] for t in tokenizer.explain('±10%')]
    assert tokens == explain_tokens