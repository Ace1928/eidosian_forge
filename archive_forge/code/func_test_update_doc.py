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
def test_update_doc(parser, model, doc, gold):
    parser.model = model

    def optimize(key, weights, gradient):
        weights -= 0.001 * gradient
        return (weights, gradient)
    example = Example.from_dict(doc, gold)
    parser.update([example], sgd=optimize)