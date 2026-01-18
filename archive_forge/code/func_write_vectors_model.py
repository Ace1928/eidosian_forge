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
def write_vectors_model(tmp_dir):
    import numpy
    vocab = Vocab()
    vector_data = {'dog': numpy.random.uniform(-1, 1, (300,)), 'cat': numpy.random.uniform(-1, 1, (300,)), 'orange': numpy.random.uniform(-1, 1, (300,))}
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    nlp_path = tmp_dir / 'vectors_model'
    nlp = English(vocab)
    nlp.to_disk(nlp_path)
    return str(nlp_path)