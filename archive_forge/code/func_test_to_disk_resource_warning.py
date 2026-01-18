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
@pytest.mark.parametrize('obj', objects_to_test[0], ids=objects_to_test[1])
def test_to_disk_resource_warning(obj):
    warnings_list = write_obj_and_catch_warnings(obj)
    assert len(warnings_list) == 0