import pytest
from thinc.api import ConfigValidationError, Linear, Model
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, combine_score_weights, registry
from ..util import make_tempdir
@pytest.mark.parametrize('i,func,func2', [(0, lambda nlp, name: lambda doc: doc, lambda doc: doc), (1, PipeFactoriesIdempotent, PipeFactoriesIdempotent(None, None))])
def test_pipe_factories_decorator_idempotent(i, func, func2):
    """Check that decorator can be run multiple times if the function is the
    same. This is especially relevant for live reloading because we don't
    want spaCy to raise an error if a module registering components is reloaded.
    """
    name = f'test_pipe_factories_decorator_idempotent_{i}'
    for i in range(5):
        Language.factory(name, func=func)
    nlp = Language()
    nlp.add_pipe(name)
    Language.factory(name, func=func)
    name2 = f'{name}2'
    for i in range(5):
        Language.component(name2, func=func2)
    nlp = Language()
    nlp.add_pipe(name)
    Language.component(name2, func=func2)