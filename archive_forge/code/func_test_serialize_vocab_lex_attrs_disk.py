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
@pytest.mark.parametrize('strings,lex_attr', test_strings_attrs)
def test_serialize_vocab_lex_attrs_disk(strings, lex_attr):
    vocab1 = Vocab(strings=strings)
    vocab2 = Vocab()
    vocab1[strings[0]].norm_ = lex_attr
    assert vocab1[strings[0]].norm_ == lex_attr
    assert vocab2[strings[0]].norm_ != lex_attr
    with make_tempdir() as d:
        file_path = d / 'vocab'
        vocab1.to_disk(file_path)
        vocab2 = vocab2.from_disk(file_path)
    assert vocab2[strings[0]].norm_ == lex_attr