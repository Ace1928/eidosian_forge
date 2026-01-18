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
def write_sample_jsonl(tmp_dir):
    data = [{'meta': {'id': '1'}, 'text': "This is the best TV you'll ever buy!", 'cats': {'pos': 1, 'neg': 0}}, {'meta': {'id': '2'}, 'text': "I wouldn't buy this again.", 'cats': {'pos': 0, 'neg': 1}}]
    file_path = f'{tmp_dir}/text.jsonl'
    srsly.write_jsonl(file_path, data)
    return file_path