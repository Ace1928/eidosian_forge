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
def test_tok2vec_listener_overfitting():
    """Test that a pipeline with a listener properly overfits, even if 'tok2vec' is in the annotating components"""
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses, annotates=['tok2vec'])
    assert losses['tagger'] < 1e-05
    test_text = 'I like blue eggs'
    doc = nlp(test_text)
    assert doc[0].tag_ == 'N'
    assert doc[1].tag_ == 'V'
    assert doc[2].tag_ == 'J'
    assert doc[3].tag_ == 'N'
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        assert doc2[0].tag_ == 'N'
        assert doc2[1].tag_ == 'V'
        assert doc2[2].tag_ == 'J'
        assert doc2[3].tag_ == 'N'