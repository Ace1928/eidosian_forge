import pytest
from spacy.pipeline._parser_internals import nonproj
from spacy.pipeline._parser_internals.nonproj import (
from spacy.tokens import Doc
def test_parser_contains_cycle(tree, cyclic_tree, partial_tree, multirooted_tree):
    assert contains_cycle(tree) is None
    assert contains_cycle(cyclic_tree) == {3, 4, 5}
    assert contains_cycle(partial_tree) is None
    assert contains_cycle(multirooted_tree) is None