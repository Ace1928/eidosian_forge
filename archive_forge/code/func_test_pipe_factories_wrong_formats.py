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
def test_pipe_factories_wrong_formats():
    with pytest.raises(ValueError):

        @Language.component
        def component(foo: int, bar: str):
            ...
    with pytest.raises(ValueError):

        @Language.factory
        def factory1(foo: int, bar: str):
            ...
    with pytest.raises(ValueError):

        @Language.factory('test_pipe_factories_missing_args')
        def factory2(foo: int, bar: str):
            ...