import numpy
import pytest
import srsly
from spacy.attrs import NORM
from spacy.lang.en import English
from spacy.strings import StringStore
from spacy.tokens import Doc
from spacy.vocab import Vocab
def test_pickle_doc(en_vocab):
    words = ['a', 'b', 'c']
    deps = ['dep'] * len(words)
    heads = [0] * len(words)
    doc = Doc(en_vocab, words=words, deps=deps, heads=heads)
    data = srsly.pickle_dumps(doc)
    unpickled = srsly.pickle_loads(data)
    assert [t.text for t in unpickled] == words
    assert [t.dep_ for t in unpickled] == deps
    assert [t.head.i for t in unpickled] == heads
    assert list(doc.noun_chunks) == []