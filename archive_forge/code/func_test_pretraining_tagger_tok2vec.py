from pathlib import Path
import numpy as np
import pytest
import srsly
from thinc.api import Config, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.language import DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_PRETRAIN_PATH
from spacy.ml.models.multi_task import create_pretrain_vectors
from spacy.tokens import Doc, DocBin
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from spacy.training.pretrain import pretrain
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('config', [pretrain_string_internal, pretrain_string_listener])
def test_pretraining_tagger_tok2vec(config):
    """Test pretraining of the tagger's tok2vec layer (via a listener)"""
    config = Config().from_str(pretrain_string_listener)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=False)
    filled = nlp.config
    pretrain_config = util.load_config(DEFAULT_CONFIG_PRETRAIN_PATH)
    filled = pretrain_config.merge(filled)
    with make_tempdir() as tmp_dir:
        file_path = write_sample_jsonl(tmp_dir)
        filled['paths']['raw_text'] = file_path
        filled['pretraining']['component'] = 'tagger'
        filled['pretraining']['layer'] = 'tok2vec'
        filled = filled.interpolate()
        pretrain(filled, tmp_dir)
        assert Path(tmp_dir / 'model0.bin').exists()
        assert Path(tmp_dir / 'model4.bin').exists()
        assert Path(tmp_dir / 'model-last.bin').exists()
        assert not Path(tmp_dir / 'model5.bin').exists()