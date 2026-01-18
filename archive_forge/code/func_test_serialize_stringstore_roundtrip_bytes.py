import pickle
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.strings import StringStore
from spacy.tokens import Doc
from spacy.util import ensure_path, load_model
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('strings1,strings2', test_strings)
def test_serialize_stringstore_roundtrip_bytes(strings1, strings2):
    sstore1 = StringStore(strings=strings1)
    sstore2 = StringStore(strings=strings2)
    sstore1_b = sstore1.to_bytes()
    sstore2_b = sstore2.to_bytes()
    if set(strings1) == set(strings2):
        assert sstore1_b == sstore2_b
    else:
        assert sstore1_b != sstore2_b
    sstore1 = sstore1.from_bytes(sstore1_b)
    assert sstore1.to_bytes() == sstore1_b
    new_sstore1 = StringStore().from_bytes(sstore1_b)
    assert new_sstore1.to_bytes() == sstore1_b
    assert set(new_sstore1) == set(strings1)