import pytest
from numpy.testing import assert_array_equal
from thinc.api import Config, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.ml.models.tok2vec import (
from spacy.pipeline.tok2vec import Tok2Vec, Tok2VecListener
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import registry
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_batch, make_tempdir
def test_tok2vec_listener_callback():
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ['tok2vec', 'tagger']
    tagger = nlp.get_pipe('tagger')
    tok2vec = nlp.get_pipe('tok2vec')
    docs = [nlp.make_doc('A random sentence')]
    tok2vec.model.initialize(X=docs)
    gold_array = [[1.0 for tag in ['V', 'Z']] for word in docs]
    label_sample = [tagger.model.ops.asarray(gold_array, dtype='float32')]
    tagger.model.initialize(X=docs, Y=label_sample)
    docs = [nlp.make_doc('Another entirely random sentence')]
    tok2vec.update([Example.from_dict(x, {}) for x in docs])
    Y, get_dX = tagger.model.begin_update(docs)
    assert get_dX(Y) is not None