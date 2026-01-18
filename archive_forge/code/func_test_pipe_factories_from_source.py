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
def test_pipe_factories_from_source():
    """Test adding components from a source model."""
    source_nlp = English()
    source_nlp.add_pipe('tagger', name='my_tagger')
    nlp = English()
    with pytest.raises(ValueError):
        nlp.add_pipe('my_tagger', source='en_core_web_sm')
    nlp.add_pipe('my_tagger', source=source_nlp)
    assert 'my_tagger' in nlp.pipe_names
    with pytest.raises(KeyError):
        nlp.add_pipe('custom', source=source_nlp)