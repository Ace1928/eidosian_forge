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
def test_pipe_function_component():
    name = 'test_component'

    @Language.component(name)
    def component(doc: Doc) -> Doc:
        return doc
    assert name in registry.factories
    nlp = Language()
    with pytest.raises(ValueError):
        nlp.add_pipe(component)
    nlp.add_pipe(name)
    assert name in nlp.pipe_names
    assert nlp.pipe_factories[name] == name
    assert Language.get_factory_meta(name)
    assert nlp.get_pipe_meta(name)
    pipe = nlp.get_pipe(name)
    assert pipe == component
    pipe = nlp.create_pipe(name)
    assert pipe == component