import pytest
from thinc.api import Model
from spacy import registry
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.pipeline.transition_parser import Parser
from spacy.tokens.doc import Doc
from spacy.training import Example
from spacy.vocab import Vocab
@pytest.mark.skip(reason='No longer supported')
def test_update_doc_beam(parser, model, doc, gold):
    parser.model = model

    def optimize(weights, gradient, key=None):
        weights -= 0.001 * gradient
    parser.update_beam((doc, gold), sgd=optimize)