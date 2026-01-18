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
def test_tok2vec_listener_source_link_name():
    """The component's internal name and the tok2vec listener map correspond
    to the most recently modified pipeline.
    """
    orig_config = Config().from_str(cfg_string_multi)
    nlp1 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp1.get_pipe('tok2vec').listening_components == ['tagger', 'ner']
    nlp2 = English()
    nlp2.add_pipe('tok2vec', source=nlp1)
    nlp2.add_pipe('tagger', name='tagger2', source=nlp1)
    assert nlp1.get_pipe('tagger').name == nlp2.get_pipe('tagger2').name == 'tagger2'
    assert nlp2.get_pipe('tok2vec').listening_components == ['tagger2']
    nlp2.add_pipe('ner', name='ner3', source=nlp1)
    assert nlp2.get_pipe('tok2vec').listening_components == ['tagger2', 'ner3']
    nlp2.remove_pipe('ner3')
    assert nlp2.get_pipe('tok2vec').listening_components == ['tagger2']
    nlp2.remove_pipe('tagger2')
    assert nlp2.get_pipe('tok2vec').listening_components == []
    assert nlp1.get_pipe('tok2vec').listening_components == []
    nlp1.add_pipe('sentencizer')
    assert nlp1.get_pipe('tok2vec').listening_components == ['tagger', 'ner']
    nlp2.add_pipe('sentencizer')
    assert nlp1.get_pipe('tok2vec').listening_components == []