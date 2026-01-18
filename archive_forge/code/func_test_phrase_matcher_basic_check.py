import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_basic_check(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    pattern = Doc(en_vocab, words=['hello', 'world'])
    with pytest.raises(ValueError):
        matcher.add('TEST', pattern)