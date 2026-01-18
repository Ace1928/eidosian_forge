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
def test_pipe_factories_from_source_custom():
    """Test adding components from a source model with custom components."""
    name = 'test_pipe_factories_from_source_custom'

    @Language.factory(name, default_config={'arg': 'hello'})
    def test_factory(nlp, name, arg: str):
        return lambda doc: doc
    source_nlp = English()
    source_nlp.add_pipe('tagger')
    source_nlp.add_pipe(name, config={'arg': 'world'})
    nlp = English()
    nlp.add_pipe(name, source=source_nlp)
    assert name in nlp.pipe_names
    assert nlp.get_pipe_meta(name).default_config['arg'] == 'hello'
    config = nlp.config['components'][name]
    assert config['factory'] == name
    assert config['arg'] == 'world'