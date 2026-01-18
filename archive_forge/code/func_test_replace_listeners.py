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
def test_replace_listeners():
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    examples = [Example.from_dict(nlp.make_doc('x y'), {'tags': ['V', 'Z']})]
    nlp.initialize(lambda: examples)
    tok2vec = nlp.get_pipe('tok2vec')
    tagger = nlp.get_pipe('tagger')
    assert isinstance(tagger.model.layers[0], Tok2VecListener)
    assert tok2vec.listener_map['tagger'][0] == tagger.model.layers[0]
    assert nlp.config['components']['tok2vec']['model']['@architectures'] == 'spacy.Tok2Vec.v2'
    assert nlp.config['components']['tagger']['model']['tok2vec']['@architectures'] == 'spacy.Tok2VecListener.v1'
    nlp.replace_listeners('tok2vec', 'tagger', ['model.tok2vec'])
    assert not isinstance(tagger.model.layers[0], Tok2VecListener)
    t2v_cfg = nlp.config['components']['tok2vec']['model']
    assert t2v_cfg['@architectures'] == 'spacy.Tok2Vec.v2'
    assert nlp.config['components']['tagger']['model']['tok2vec'] == t2v_cfg
    with pytest.raises(ValueError):
        nlp.replace_listeners('invalid', 'tagger', ['model.tok2vec'])
    with pytest.raises(ValueError):
        nlp.replace_listeners('tok2vec', 'parser', ['model.tok2vec'])
    with pytest.raises(ValueError):
        nlp.replace_listeners('tok2vec', 'tagger', ['model.yolo'])
    with pytest.raises(ValueError):
        nlp.replace_listeners('tok2vec', 'tagger', ['model.tok2vec', 'model.yolo'])
    optimizer = nlp.initialize(lambda: examples)
    for i in range(2):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        assert losses['tok2vec'] == 0.0
        assert losses['tagger'] > 0.0