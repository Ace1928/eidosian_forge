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
@pytest.mark.parametrize('name,textcat_config', [('textcat_multilabel', {'@architectures': 'spacy.TextCatEnsemble.v2', 'tok2vec': LAZY_INIT_TOK2VEC_MODEL, 'linear_model': {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': False, 'ngram_size': 1, 'no_output_layer': False}}), ('textcat', {'@architectures': 'spacy.TextCatEnsemble.v2', 'tok2vec': LAZY_INIT_TOK2VEC_MODEL, 'linear_model': {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': True, 'ngram_size': 5, 'no_output_layer': False}}), ('textcat', {'@architectures': 'spacy.TextCatParametricAttention.v1', 'tok2vec': LAZY_INIT_TOK2VEC_MODEL, 'exclusive_classes': True}), ('textcat_multilabel', {'@architectures': 'spacy.TextCatParametricAttention.v1', 'tok2vec': LAZY_INIT_TOK2VEC_MODEL, 'exclusive_classes': False}), ('textcat', {'@architectures': 'spacy.TextCatReduce.v1', 'tok2vec': LAZY_INIT_TOK2VEC_MODEL, 'exclusive_classes': True, 'use_reduce_first': True, 'use_reduce_last': True, 'use_reduce_max': True, 'use_reduce_mean': True}), ('textcat_multilabel', {'@architectures': 'spacy.TextCatReduce.v1', 'tok2vec': LAZY_INIT_TOK2VEC_MODEL, 'exclusive_classes': False, 'use_reduce_first': True, 'use_reduce_last': True, 'use_reduce_max': True, 'use_reduce_mean': True})])
def test_tok2vec_lazy_init(name, textcat_config):
    nlp = English()
    pipe_config = {'model': textcat_config}
    textcat = nlp.add_pipe(name, config=pipe_config)
    textcat.add_label('POSITIVE')
    textcat.add_label('NEGATIVE')
    nlp.initialize()
    nlp.pipe(['This is a test.'])