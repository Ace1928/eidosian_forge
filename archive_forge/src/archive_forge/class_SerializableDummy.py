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
class SerializableDummy:

    def __init__(self, **cfg):
        if cfg:
            self.cfg = cfg
        else:
            self.cfg = None
        super(SerializableDummy, self).__init__()

    def to_bytes(self, exclude=tuple(), disable=None, **kwargs):
        return srsly.msgpack_dumps({'dummy': srsly.json_dumps(None)})

    def from_bytes(self, bytes_data, exclude):
        return self

    def to_disk(self, path, exclude=tuple(), **kwargs):
        pass

    def from_disk(self, path, exclude=tuple(), **kwargs):
        return self