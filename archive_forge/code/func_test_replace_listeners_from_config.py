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
def test_replace_listeners_from_config():
    orig_config = Config().from_str(cfg_string_multi)
    nlp = util.load_model_from_config(orig_config, auto_fill=True)
    annots = {'tags': ['V', 'Z'], 'entities': [(0, 1, 'A'), (1, 2, 'B')]}
    examples = [Example.from_dict(nlp.make_doc('x y'), annots)]
    nlp.initialize(lambda: examples)
    tok2vec = nlp.get_pipe('tok2vec')
    tagger = nlp.get_pipe('tagger')
    ner = nlp.get_pipe('ner')
    assert tok2vec.listening_components == ['tagger', 'ner']
    assert any((isinstance(node, Tok2VecListener) for node in ner.model.walk()))
    assert any((isinstance(node, Tok2VecListener) for node in tagger.model.walk()))
    with make_tempdir() as dir_path:
        nlp.to_disk(dir_path)
        base_model = str(dir_path)
        new_config = {'nlp': {'lang': 'en', 'pipeline': ['tok2vec', 'tagger2', 'ner3', 'tagger4']}, 'components': {'tok2vec': {'source': base_model}, 'tagger2': {'source': base_model, 'component': 'tagger', 'replace_listeners': ['model.tok2vec']}, 'ner3': {'source': base_model, 'component': 'ner'}, 'tagger4': {'source': base_model, 'component': 'tagger'}}}
        new_nlp = util.load_model_from_config(new_config, auto_fill=True)
    new_nlp.initialize(lambda: examples)
    tok2vec = new_nlp.get_pipe('tok2vec')
    tagger = new_nlp.get_pipe('tagger2')
    ner = new_nlp.get_pipe('ner3')
    assert 'ner' not in new_nlp.pipe_names
    assert 'tagger' not in new_nlp.pipe_names
    assert tok2vec.listening_components == ['ner3', 'tagger4']
    assert any((isinstance(node, Tok2VecListener) for node in ner.model.walk()))
    assert not any((isinstance(node, Tok2VecListener) for node in tagger.model.walk()))
    t2v_cfg = new_nlp.config['components']['tok2vec']['model']
    assert t2v_cfg['@architectures'] == 'spacy.Tok2Vec.v2'
    assert new_nlp.config['components']['tagger2']['model']['tok2vec'] == t2v_cfg
    assert new_nlp.config['components']['ner3']['model']['tok2vec']['@architectures'] == 'spacy.Tok2VecListener.v1'
    assert new_nlp.config['components']['tagger4']['model']['tok2vec']['@architectures'] == 'spacy.Tok2VecListener.v1'