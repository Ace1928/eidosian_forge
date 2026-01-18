import random
import numpy.random
import pytest
from numpy.testing import assert_almost_equal
from thinc.api import Config, compounding, fix_random_seed, get_current_ops
from wasabi import msg
import spacy
from spacy import util
from spacy.cli.evaluate import print_prf_per_type, print_textcats_auc_per_cat
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import TextCategorizer
from spacy.pipeline.textcat import (
from spacy.pipeline.textcat_multilabel import (
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.training.initialize import init_nlp
from ..tok2vec import build_lazy_init_tok2vec as _
from ..util import make_tempdir
def test_textcat_multilabel_threshold():
    nlp = English()
    nlp.add_pipe('textcat_multilabel')
    train_examples = []
    for text, annotations in TRAIN_DATA_SINGLE_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    nlp.initialize(get_examples=lambda: train_examples)
    scores = nlp.evaluate(train_examples)
    assert 0 <= scores['cats_score'] <= 1
    scores = nlp.evaluate(train_examples, scorer_cfg={'threshold': 1.0})
    assert scores['cats_f_per_type']['POSITIVE']['r'] == 0
    scores = nlp.evaluate(train_examples, scorer_cfg={'threshold': 0})
    macro_f = scores['cats_score']
    assert scores['cats_f_per_type']['POSITIVE']['r'] == 1.0
    scores = nlp.evaluate(train_examples, scorer_cfg={'threshold': 0, 'positive_label': 'POSITIVE'})
    pos_f = scores['cats_score']
    assert scores['cats_f_per_type']['POSITIVE']['r'] == 1.0
    assert pos_f >= macro_f