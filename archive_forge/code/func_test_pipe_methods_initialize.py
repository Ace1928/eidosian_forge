import gc
import numpy
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.lang.en.syntax_iterators import noun_chunks
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList, get_arg_names, make_tempdir
from spacy.vocab import Vocab
def test_pipe_methods_initialize():
    """Test that the [initialize] config reflects the components correctly."""
    nlp = Language()
    nlp.add_pipe('tagger')
    assert 'tagger' not in nlp.config['initialize']['components']
    nlp.config['initialize']['components']['tagger'] = {'labels': ['hello']}
    assert nlp.config['initialize']['components']['tagger'] == {'labels': ['hello']}
    nlp.remove_pipe('tagger')
    assert 'tagger' not in nlp.config['initialize']['components']
    nlp.add_pipe('tagger')
    assert 'tagger' not in nlp.config['initialize']['components']
    nlp.config['initialize']['components']['tagger'] = {'labels': ['hello']}
    nlp.rename_pipe('tagger', 'my_tagger')
    assert 'tagger' not in nlp.config['initialize']['components']
    assert nlp.config['initialize']['components']['my_tagger'] == {'labels': ['hello']}
    nlp.config['initialize']['components']['test'] = {'foo': 'bar'}
    nlp.add_pipe('ner', name='test')
    assert 'test' in nlp.config['initialize']['components']
    nlp.remove_pipe('test')
    assert 'test' not in nlp.config['initialize']['components']