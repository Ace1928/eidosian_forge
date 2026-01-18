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
def test_warning_pipe_begin_training():
    with pytest.warns(UserWarning, match='begin_training'):

        class IncompatPipe(TrainablePipe):

            def __init__(self):
                ...

            def begin_training(*args, **kwargs):
                ...