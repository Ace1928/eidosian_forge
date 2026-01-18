import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_pickle(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    mock = Mock()
    matcher.add('TEST', [Doc(en_vocab, words=['test'])])
    matcher.add('TEST2', [Doc(en_vocab, words=['test2'])], on_match=mock)
    doc = Doc(en_vocab, words=['these', 'are', 'tests', ':', 'test', 'test2'])
    assert len(matcher) == 2
    b = srsly.pickle_dumps(matcher)
    matcher_unpickled = srsly.pickle_loads(b)
    matches = matcher(doc)
    matches_unpickled = matcher_unpickled(doc)
    assert len(matcher) == len(matcher_unpickled)
    assert matches == matches_unpickled
    vocab, docs, callbacks, attr = matcher_unpickled.__reduce__()[1]
    assert isinstance(callbacks.get('TEST2'), Mock)