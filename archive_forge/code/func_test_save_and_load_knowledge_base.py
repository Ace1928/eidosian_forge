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
def test_save_and_load_knowledge_base():
    nlp = Language()
    kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    with make_tempdir() as d:
        path = d / 'kb'
        try:
            kb.to_disk(path)
        except Exception as e:
            pytest.fail(str(e))
        try:
            kb_loaded = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
            kb_loaded.from_disk(path)
        except Exception as e:
            pytest.fail(str(e))