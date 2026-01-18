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
def test_pipe_factory_meta_config_cleanup():
    """Test that component-specific meta and config entries are represented
    correctly and cleaned up when pipes are removed, replaced or renamed."""
    nlp = Language()
    nlp.add_pipe('ner', name='ner_component')
    nlp.add_pipe('textcat')
    assert nlp.get_factory_meta('ner')
    assert nlp.get_pipe_meta('ner_component')
    assert nlp.get_pipe_config('ner_component')
    assert nlp.get_factory_meta('textcat')
    assert nlp.get_pipe_meta('textcat')
    assert nlp.get_pipe_config('textcat')
    nlp.rename_pipe('textcat', 'tc')
    assert nlp.get_pipe_meta('tc')
    assert nlp.get_pipe_config('tc')
    with pytest.raises(ValueError):
        nlp.remove_pipe('ner')
    nlp.remove_pipe('ner_component')
    assert 'ner_component' not in nlp._pipe_meta
    assert 'ner_component' not in nlp._pipe_configs
    with pytest.raises(ValueError):
        nlp.replace_pipe('textcat', 'parser')
    nlp.replace_pipe('tc', 'parser')
    assert nlp.get_factory_meta('parser')
    assert nlp.get_pipe_meta('tc').factory == 'parser'