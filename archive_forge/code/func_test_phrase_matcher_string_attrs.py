import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_string_attrs(en_vocab):
    words1 = ['I', 'like', 'cats']
    pos1 = ['PRON', 'VERB', 'NOUN']
    words2 = ['Yes', ',', 'you', 'hate', 'dogs', 'very', 'much']
    pos2 = ['INTJ', 'PUNCT', 'PRON', 'VERB', 'NOUN', 'ADV', 'ADV']
    pattern = Doc(en_vocab, words=words1, pos=pos1)
    matcher = PhraseMatcher(en_vocab, attr='POS')
    matcher.add('TEST', [pattern])
    doc = Doc(en_vocab, words=words2, pos=pos2)
    matches = matcher(doc)
    assert len(matches) == 1
    match_id, start, end = matches[0]
    assert match_id == en_vocab.strings['TEST']
    assert start == 2
    assert end == 5