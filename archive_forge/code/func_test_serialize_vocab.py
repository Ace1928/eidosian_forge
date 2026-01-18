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
@pytest.mark.parametrize('text', ['rat'])
def test_serialize_vocab(en_vocab, text):
    text_hash = en_vocab.strings.add(text)
    vocab_bytes = en_vocab.to_bytes(exclude=['lookups'])
    new_vocab = Vocab().from_bytes(vocab_bytes)
    assert new_vocab.strings[text_hash] == text
    assert new_vocab.to_bytes(exclude=['lookups']) == vocab_bytes