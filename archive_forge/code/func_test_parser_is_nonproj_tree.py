import pytest
from spacy.pipeline._parser_internals import nonproj
from spacy.pipeline._parser_internals.nonproj import (
from spacy.tokens import Doc
def test_parser_is_nonproj_tree(proj_tree, cyclic_tree, nonproj_tree, partial_tree, multirooted_tree):
    assert is_nonproj_tree(proj_tree) is False
    assert is_nonproj_tree(nonproj_tree) is True
    assert is_nonproj_tree(partial_tree) is False
    assert is_nonproj_tree(multirooted_tree) is True
    with pytest.raises(ValueError, match='Found cycle in dependency graph: \\[1, 2, 2, 4, 5, 3, 2\\]'):
        is_nonproj_tree(cyclic_tree)