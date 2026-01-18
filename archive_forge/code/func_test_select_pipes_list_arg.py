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
def test_select_pipes_list_arg(nlp):
    for name in ['c1', 'c2', 'c3']:
        nlp.add_pipe('new_pipe', name=name)
        assert nlp.has_pipe(name)
    with nlp.select_pipes(disable=['c1', 'c2']):
        assert not nlp.has_pipe('c1')
        assert not nlp.has_pipe('c2')
        assert nlp.has_pipe('c3')
    with nlp.select_pipes(enable='c3'):
        assert not nlp.has_pipe('c1')
        assert not nlp.has_pipe('c2')
        assert nlp.has_pipe('c3')
    with nlp.select_pipes(enable=['c1', 'c2'], disable='c3'):
        assert nlp.has_pipe('c1')
        assert nlp.has_pipe('c2')
        assert not nlp.has_pipe('c3')
    with nlp.select_pipes(enable=[]):
        assert not nlp.has_pipe('c1')
        assert not nlp.has_pipe('c2')
        assert not nlp.has_pipe('c3')
    with nlp.select_pipes(enable=['c1', 'c2', 'c3'], disable=[]):
        assert nlp.has_pipe('c1')
        assert nlp.has_pipe('c2')
        assert nlp.has_pipe('c3')
    with nlp.select_pipes(disable=['c1', 'c2', 'c3'], enable=[]):
        assert not nlp.has_pipe('c1')
        assert not nlp.has_pipe('c2')
        assert not nlp.has_pipe('c3')