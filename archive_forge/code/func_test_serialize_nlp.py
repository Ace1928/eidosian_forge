import pytest
from catalogue import RegistryError
from thinc.api import Config, ConfigValidationError
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import DEFAULT_CONFIG, DEFAULT_CONFIG_PRETRAIN_PATH, Language
from spacy.ml.models import (
from spacy.schemas import ConfigSchema, ConfigSchemaPretrain
from spacy.training import Example
from spacy.util import (
from ..util import make_tempdir
def test_serialize_nlp():
    """Create a custom nlp pipeline from config and ensure it serializes it correctly"""
    nlp_config = Config().from_str(nlp_config_string)
    nlp = load_model_from_config(nlp_config, auto_fill=True)
    nlp.get_pipe('tagger').add_label('A')
    nlp.initialize()
    assert 'tok2vec' in nlp.pipe_names
    assert 'tagger' in nlp.pipe_names
    assert 'parser' not in nlp.pipe_names
    assert nlp.get_pipe('tagger').model.get_ref('tok2vec').get_dim('nO') == 342
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        assert 'tok2vec' in nlp2.pipe_names
        assert 'tagger' in nlp2.pipe_names
        assert 'parser' not in nlp2.pipe_names
        assert nlp2.get_pipe('tagger').model.get_ref('tok2vec').get_dim('nO') == 342