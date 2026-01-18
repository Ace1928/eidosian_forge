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
def test_tokenizer_flush_cache(en_vocab):
    suffix_re = re.compile('[\\.]$')
    tokenizer = Tokenizer(en_vocab, suffix_search=suffix_re.search)
    assert [t.text for t in tokenizer('a.')] == ['a', '.']
    tokenizer.suffix_search = None
    assert [t.text for t in tokenizer('a.')] == ['a.']