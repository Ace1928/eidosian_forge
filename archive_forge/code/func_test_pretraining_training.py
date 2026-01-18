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
def test_pretraining_training():
    """Test that training can use a pretrained Tok2Vec model"""
    config = Config().from_str(pretrain_string_internal)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=False)
    filled = nlp.config
    pretrain_config = util.load_config(DEFAULT_CONFIG_PRETRAIN_PATH)
    filled = pretrain_config.merge(filled)
    train_config = util.load_config(DEFAULT_CONFIG_PATH)
    filled = train_config.merge(filled)
    with make_tempdir() as tmp_dir:
        pretrain_dir = tmp_dir / 'pretrain'
        pretrain_dir.mkdir()
        file_path = write_sample_jsonl(pretrain_dir)
        filled['paths']['raw_text'] = file_path
        filled['pretraining']['component'] = 'tagger'
        filled['pretraining']['layer'] = 'tok2vec'
        train_dir = tmp_dir / 'train'
        train_dir.mkdir()
        train_path, dev_path = write_sample_training(train_dir)
        filled['paths']['train'] = train_path
        filled['paths']['dev'] = dev_path
        filled = filled.interpolate()
        P = filled['pretraining']
        nlp_base = init_nlp(filled)
        model_base = nlp_base.get_pipe(P['component']).model.get_ref(P['layer']).get_ref('embed')
        embed_base = None
        for node in model_base.walk():
            if node.name == 'hashembed':
                embed_base = node
        pretrain(filled, pretrain_dir)
        pretrained_model = Path(pretrain_dir / 'model3.bin')
        assert pretrained_model.exists()
        filled['initialize']['init_tok2vec'] = str(pretrained_model)
        nlp = init_nlp(filled)
        model = nlp.get_pipe(P['component']).model.get_ref(P['layer']).get_ref('embed')
        embed = None
        for node in model.walk():
            if node.name == 'hashembed':
                embed = node
        assert np.any(np.not_equal(embed.get_param('E'), embed_base.get_param('E')))
        train(nlp, train_dir)