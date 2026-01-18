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
def test_pipe_factories_language_specific():
    """Test that language sub-classes can have their own factories, with
    fallbacks to the base factories."""
    name1 = 'specific_component1'
    name2 = 'specific_component2'
    Language.component(name1, func=lambda: 'base')
    English.component(name1, func=lambda: 'en')
    German.component(name2, func=lambda: 'de')
    assert Language.has_factory(name1)
    assert not Language.has_factory(name2)
    assert English.has_factory(name1)
    assert not English.has_factory(name2)
    assert German.has_factory(name1)
    assert German.has_factory(name2)
    nlp = Language()
    assert nlp.create_pipe(name1)() == 'base'
    with pytest.raises(ValueError):
        nlp.create_pipe(name2)
    nlp_en = English()
    assert nlp_en.create_pipe(name1)() == 'en'
    with pytest.raises(ValueError):
        nlp_en.create_pipe(name2)
    nlp_de = German()
    assert nlp_de.create_pipe(name1)() == 'base'
    assert nlp_de.create_pipe(name2)() == 'de'