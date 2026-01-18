import pytest
from spacy.tokens import Doc
from ..util import apply_transition_sequence
def test_parser_space_attachment(en_vocab):
    words = ['This', 'is', 'a', 'test', '.', '\n', 'To', 'ensure', ' ', 'spaces', 'are', 'attached', 'well', '.']
    heads = [1, 1, 3, 1, 1, 4, 7, 11, 7, 11, 11, 11, 11, 11]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    for sent in doc.sents:
        if len(sent) == 1:
            assert not sent[-1].is_space