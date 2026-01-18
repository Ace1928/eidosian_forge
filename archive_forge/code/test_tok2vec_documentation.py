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
The component's internal name and the tok2vec listener map correspond
    to the most recently modified pipeline.
    