import pytest
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
def test_stack_depth(doc):
    state = StateClass(doc)
    assert state.stack_depth() == 0
    assert state.buffer_length() == len(doc)
    state.push()
    assert state.buffer_length() == 3
    assert state.stack_depth() == 1