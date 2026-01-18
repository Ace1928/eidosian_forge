import pytest
from spacy.tokens import Doc
def test_parser_parse_navigate_child_consistency(en_vocab, words, heads):
    doc = Doc(en_vocab, words=words, heads=heads, deps=['dep'] * len(heads))
    lefts = {}
    rights = {}
    for head in doc:
        assert head.i not in lefts
        lefts[head.i] = set()
        for left in head.lefts:
            lefts[head.i].add(left.i)
        assert head.i not in rights
        rights[head.i] = set()
        for right in head.rights:
            rights[head.i].add(right.i)
    for head in doc:
        assert head.n_rights == len(rights[head.i])
        assert head.n_lefts == len(lefts[head.i])
    for child in doc:
        if child.i < child.head.i:
            assert child.i in lefts[child.head.i]
            assert child.i not in rights[child.head.i]
            lefts[child.head.i].remove(child.i)
        elif child.i > child.head.i:
            assert child.i in rights[child.head.i]
            assert child.i not in lefts[child.head.i]
            rights[child.head.i].remove(child.i)
    for head_index, children in lefts.items():
        assert not children
    for head_index, children in rights.items():
        assert not children