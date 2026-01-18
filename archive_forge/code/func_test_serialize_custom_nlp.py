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
def test_serialize_custom_nlp():
    """Create a custom nlp pipeline and ensure it serializes it correctly"""
    nlp = English()
    parser_cfg = dict()
    parser_cfg['model'] = {'@architectures': 'my_test_parser'}
    nlp.add_pipe('parser', config=parser_cfg)
    nlp.initialize()
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        model = nlp2.get_pipe('parser').model
        model.get_ref('tok2vec')
        assert model.get_ref('upper').get_dim('nI') == 65
        assert model.get_ref('lower').get_dim('nI') == 65