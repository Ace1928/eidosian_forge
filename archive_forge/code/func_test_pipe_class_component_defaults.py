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
def test_pipe_class_component_defaults():
    name = 'test_class_component_defaults'

    @Language.factory(name)
    class Component:

        def __init__(self, nlp: Language, name: str, value1: StrictInt=StrictInt(10), value2: StrictStr=StrictStr('hello')):
            self.nlp = nlp
            self.value1 = value1
            self.value2 = value2

        def __call__(self, doc: Doc) -> Doc:
            return doc
    nlp = Language()
    nlp.add_pipe(name)
    pipe = nlp.get_pipe(name)
    assert isinstance(pipe.nlp, Language)
    assert pipe.value1 == 10
    assert pipe.value2 == 'hello'