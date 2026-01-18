import pytest
from spacy.tokens import Doc
def test_parser_parse_navigate_consistency(en_vocab, words, heads):
    doc = Doc(en_vocab, words=words, heads=heads, deps=['dep'] * len(heads))
    for head in doc:
        for child in head.lefts:
            assert child.head == head
        for child in head.rights:
            assert child.head == head