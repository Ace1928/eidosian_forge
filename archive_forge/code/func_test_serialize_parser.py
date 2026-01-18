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
@pytest.mark.parametrize('parser_config_string', [parser_config_string_upper, parser_config_string_no_upper])
def test_serialize_parser(parser_config_string):
    """Create a non-default parser config to check nlp serializes it correctly"""
    nlp = English()
    model_config = Config().from_str(parser_config_string)
    parser = nlp.add_pipe('parser', config=model_config)
    parser.add_label('nsubj')
    nlp.initialize()
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        model = nlp2.get_pipe('parser').model
        model.get_ref('tok2vec')
        if model.attrs['has_upper']:
            assert model.get_ref('upper').get_dim('nI') == 66
        assert model.get_ref('lower').get_dim('nI') == 66