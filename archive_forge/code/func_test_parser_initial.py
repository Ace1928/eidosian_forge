import pytest
from numpy.testing import assert_equal
from thinc.api import Adam
from spacy import registry, util
from spacy.attrs import DEP, NORM
from spacy.lang.en import English
from spacy.pipeline import DependencyParser
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from ..util import apply_transition_sequence, make_tempdir
@pytest.mark.skip(reason='The step_through API was removed (but should be brought back)')
def test_parser_initial(en_vocab, en_parser):
    words = ['I', 'ate', 'the', 'pizza', 'with', 'anchovies', '.']
    transition = ['L-nsubj', 'S', 'L-det']
    doc = Doc(en_vocab, words=words)
    apply_transition_sequence(en_parser, doc, transition)
    assert doc[0].head.i == 1
    assert doc[1].head.i == 1
    assert doc[2].head.i == 3
    assert doc[3].head.i == 3