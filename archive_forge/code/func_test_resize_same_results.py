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
@pytest.mark.parametrize('name,textcat_config', [('textcat', {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': True, 'no_output_layer': False, 'ngram_size': 3}), ('textcat', {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': True, 'no_output_layer': True, 'ngram_size': 3}), ('textcat_multilabel', {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': False, 'no_output_layer': False, 'ngram_size': 3}), ('textcat_multilabel', {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': False, 'no_output_layer': True, 'ngram_size': 3}), ('textcat', {'@architectures': 'spacy.TextCatReduce.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': True, 'use_reduce_first': True, 'use_reduce_last': True, 'use_reduce_max': True, 'use_reduce_mean': True}), ('textcat_multilabel', {'@architectures': 'spacy.TextCatReduce.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': False, 'use_reduce_first': True, 'use_reduce_last': True, 'use_reduce_max': True, 'use_reduce_mean': True})])
def test_resize_same_results(name, textcat_config):
    fix_random_seed(0)
    nlp = English()
    pipe_config = {'model': textcat_config}
    textcat = nlp.add_pipe(name, config=pipe_config)
    train_examples = []
    for text, annotations in TRAIN_DATA_SINGLE_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    assert textcat.model.maybe_get_dim('nO') in [2, None]
    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    test_text = 'I am happy.'
    doc = nlp(test_text)
    assert len(doc.cats) == 2
    pos_pred = doc.cats['POSITIVE']
    neg_pred = doc.cats['NEGATIVE']
    textcat.add_label('NEUTRAL')
    doc = nlp(test_text)
    assert len(doc.cats) == 3
    assert doc.cats['POSITIVE'] == pos_pred
    assert doc.cats['NEGATIVE'] == neg_pred
    assert doc.cats['NEUTRAL'] <= 1
    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    doc = nlp(test_text)
    assert len(doc.cats) == 3
    assert doc.cats['POSITIVE'] != pos_pred
    assert doc.cats['NEGATIVE'] != neg_pred
    for cat in doc.cats:
        assert doc.cats[cat] <= 1