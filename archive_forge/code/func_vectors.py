import warnings
from unittest import TestCase
import pytest
import srsly
from numpy import zeros
from spacy.kb.kb_in_memory import InMemoryLookupKB, Writer
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
def vectors():
    data = zeros((3, 1), dtype='f')
    keys = ['cat', 'dog', 'rat']
    return Vectors(data=data, keys=keys)