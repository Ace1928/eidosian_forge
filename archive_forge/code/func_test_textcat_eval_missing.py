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
@pytest.mark.parametrize('multi_label,spring_p', [(True, 1 / 1), (False, 1 / 2)])
def test_textcat_eval_missing(multi_label: bool, spring_p: float):
    """
    multi-label: the missing 'spring' in gold_doc_2 doesn't incur a penalty
    exclusive labels: the missing 'spring' in gold_doc_2 is interpreted as 0.0"""
    train_examples = []
    nlp = English()
    ref1 = nlp('one')
    ref1.cats = {'winter': 0.0, 'summer': 0.0, 'autumn': 0.0, 'spring': 1.0}
    pred1 = nlp('one')
    pred1.cats = {'winter': 0.0, 'summer': 0.0, 'autumn': 0.0, 'spring': 1.0}
    train_examples.append(Example(ref1, pred1))
    ref2 = nlp('two')
    ref2.cats = {'winter': 0.0, 'summer': 0.0, 'autumn': 1.0}
    pred2 = nlp('two')
    pred2.cats = {'winter': 0.0, 'summer': 0.0, 'autumn': 0.0, 'spring': 1.0}
    train_examples.append(Example(pred2, ref2))
    scores = Scorer().score_cats(train_examples, 'cats', labels=['winter', 'summer', 'spring', 'autumn'], multi_label=multi_label)
    assert scores['cats_f_per_type']['spring']['p'] == spring_p
    assert scores['cats_f_per_type']['spring']['r'] == 1 / 1