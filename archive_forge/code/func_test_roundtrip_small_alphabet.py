import pickle
import hypothesis.strategies as st
import pytest
from hypothesis import given
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline._edit_tree_internals.edit_trees import EditTrees
from spacy.strings import StringStore
from spacy.training import Example
from spacy.util import make_tempdir
@given(st.text(alphabet='ab'), st.text(alphabet='ab'))
def test_roundtrip_small_alphabet(form, lemma):
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma