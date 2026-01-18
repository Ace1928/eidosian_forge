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
def test_pipe_factories_empty_dict_default():
    """Test that default config values can be empty dicts and that no config
    validation error is raised."""
    name = 'test_pipe_factories_empty_dict_default'

    @Language.factory(name, default_config={'foo': {}})
    def factory(nlp: Language, name: str, foo: dict):
        ...
    nlp = Language()
    nlp.create_pipe(name)