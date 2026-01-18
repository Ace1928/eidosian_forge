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
@pytest.mark.parametrize('multi_label,expected_loss', [(True, 0), (False, 0.125)])
def test_textcat_loss(multi_label: bool, expected_loss: float):
    """
    multi-label: the missing 'spring' in gold_doc_2 doesn't incur an increase in loss
    exclusive labels: the missing 'spring' in gold_doc_2 is interpreted as 0.0 and adds to the loss"""
    train_examples = []
    nlp = English()
    doc1 = nlp('one')
    cats1 = {'winter': 0.0, 'summer': 0.0, 'autumn': 0.0, 'spring': 1.0}
    train_examples.append(Example.from_dict(doc1, {'cats': cats1}))
    doc2 = nlp('two')
    cats2 = {'winter': 0.0, 'summer': 0.0, 'autumn': 1.0}
    train_examples.append(Example.from_dict(doc2, {'cats': cats2}))
    if multi_label:
        textcat = nlp.add_pipe('textcat_multilabel')
    else:
        textcat = nlp.add_pipe('textcat')
    assert isinstance(textcat, TextCategorizer)
    textcat.initialize(lambda: train_examples)
    scores = textcat.model.ops.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype='f')
    loss, d_scores = textcat.get_loss(train_examples, scores)
    assert loss == expected_loss