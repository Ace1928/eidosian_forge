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
def test_unapplicable_trees():
    strings = StringStore()
    trees = EditTrees(strings)
    tree3 = trees.add('deelt', 'delen')
    assert trees.apply(tree3, 'deeld') == None
    assert trees.apply(tree3, 'de') == None