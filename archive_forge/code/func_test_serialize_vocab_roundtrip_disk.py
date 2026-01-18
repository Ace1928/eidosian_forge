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
def test_serialize_vocab_roundtrip_disk(strings1, strings2):
    vocab1 = Vocab(strings=strings1)
    vocab2 = Vocab(strings=strings2)
    with make_tempdir() as d:
        file_path1 = d / 'vocab1'
        file_path2 = d / 'vocab2'
        vocab1.to_disk(file_path1)
        vocab2.to_disk(file_path2)
        vocab1_d = Vocab().from_disk(file_path1)
        vocab2_d = Vocab().from_disk(file_path2)
        assert set(strings1) == set([s for s in vocab1_d.strings])
        assert set(strings2) == set([s for s in vocab2_d.strings])
        if set(strings1) == set(strings2):
            assert [s for s in vocab1_d.strings] == [s for s in vocab2_d.strings]
        else:
            assert [s for s in vocab1_d.strings] != [s for s in vocab2_d.strings]