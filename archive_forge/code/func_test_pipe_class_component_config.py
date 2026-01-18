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
def test_pipe_class_component_config():
    name = 'test_class_component_config'

    @Language.factory(name)
    class Component:

        def __init__(self, nlp: Language, name: str, value1: StrictInt, value2: StrictStr):
            self.nlp = nlp
            self.value1 = value1
            self.value2 = value2
            self.is_base = True
            self.name = name

        def __call__(self, doc: Doc) -> Doc:
            return doc

    @English.factory(name)
    class ComponentEN:

        def __init__(self, nlp: Language, name: str, value1: StrictInt, value2: StrictStr):
            self.nlp = nlp
            self.value1 = value1
            self.value2 = value2
            self.is_base = False

        def __call__(self, doc: Doc) -> Doc:
            return doc
    nlp = Language()
    with pytest.raises(ConfigValidationError):
        nlp.add_pipe(name)
    with pytest.raises(ConfigValidationError):
        nlp.add_pipe(name, config={'value1': '10', 'value2': 'hello'})
    with pytest.warns(UserWarning):
        nlp.add_pipe(name, config={'value1': 10, 'value2': 'hello', 'name': 'wrong_name'})
    pipe = nlp.get_pipe(name)
    assert isinstance(pipe.nlp, Language)
    assert pipe.value1 == 10
    assert pipe.value2 == 'hello'
    assert pipe.is_base is True
    assert pipe.name == name
    nlp_en = English()
    with pytest.raises(ConfigValidationError):
        nlp_en.add_pipe(name, config={'value1': '10', 'value2': 'hello'})
    nlp_en.add_pipe(name, config={'value1': 10, 'value2': 'hello'})
    pipe = nlp_en.get_pipe(name)
    assert isinstance(pipe.nlp, English)
    assert pipe.value1 == 10
    assert pipe.value2 == 'hello'
    assert pipe.is_base is False