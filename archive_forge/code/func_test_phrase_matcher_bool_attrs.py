import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_bool_attrs(en_vocab):
    words1 = ['Hello', 'world', '!']
    words2 = ['No', 'problem', ',', 'he', 'said', '.']
    pattern = Doc(en_vocab, words=words1)
    matcher = PhraseMatcher(en_vocab, attr='IS_PUNCT')
    matcher.add('TEST', [pattern])
    doc = Doc(en_vocab, words=words2)
    matches = matcher(doc)
    assert len(matches) == 2
    match_id1, start1, end1 = matches[0]
    match_id2, start2, end2 = matches[1]
    assert match_id1 == en_vocab.strings['TEST']
    assert match_id2 == en_vocab.strings['TEST']
    assert start1 == 0
    assert end1 == 3
    assert start2 == 3
    assert end2 == 6