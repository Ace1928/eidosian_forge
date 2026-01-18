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
@pytest.mark.parametrize('component', ['ner', 'tagger', 'parser', 'textcat'])
def test_pipe_base_class_add_label(nlp, component):
    label = 'TEST'
    pipe = nlp.create_pipe(component)
    pipe.add_label(label)
    if component == 'tagger':
        assert label in pipe.labels
    else:
        assert pipe.labels == (label,)