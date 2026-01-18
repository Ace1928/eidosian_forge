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
@pytest.mark.parametrize('words', [['Hello']])
def test_parser_parse_one_word_sentence(en_vocab, en_parser, words):
    doc = Doc(en_vocab, words=words, heads=[0], deps=['ROOT'])
    assert len(doc) == 1
    with en_parser.step_through(doc) as _:
        pass
    assert doc[0].dep != 0