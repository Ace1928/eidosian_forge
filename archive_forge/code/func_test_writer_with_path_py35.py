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
def test_writer_with_path_py35():
    writer = None
    with make_tempdir() as d:
        path = d / 'test'
        try:
            writer = Writer(path)
        except Exception as e:
            pytest.fail(str(e))
        finally:
            if writer:
                writer.close()