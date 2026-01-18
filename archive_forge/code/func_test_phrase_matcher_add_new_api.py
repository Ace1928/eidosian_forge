import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_add_new_api(en_vocab):
    doc = Doc(en_vocab, words=['a', 'b'])
    patterns = [Doc(en_vocab, words=['a']), Doc(en_vocab, words=['a', 'b'])]
    matcher = PhraseMatcher(en_vocab)
    matcher.add('OLD_API', None, *patterns)
    assert len(matcher(doc)) == 2
    matcher = PhraseMatcher(en_vocab)
    on_match = Mock()
    matcher.add('OLD_API_CALLBACK', on_match, *patterns)
    assert len(matcher(doc)) == 2
    assert on_match.call_count == 2
    matcher = PhraseMatcher(en_vocab)
    matcher.add('NEW_API', patterns)
    assert len(matcher(doc)) == 2
    matcher = PhraseMatcher(en_vocab)
    on_match = Mock()
    matcher.add('NEW_API_CALLBACK', patterns, on_match=on_match)
    assert len(matcher(doc)) == 2
    assert on_match.call_count == 2