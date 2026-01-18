import os
import pytest
from spacy.attrs import IS_ALPHA, LEMMA, ORTH
from spacy.lang.en import English
from spacy.parts_of_speech import NOUN, VERB
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_to_disk():
    nlp = English()
    with make_tempdir() as d:
        nlp.vocab.to_disk(d)
        assert 'vectors' in os.listdir(d)
        assert 'lookups.bin' in os.listdir(d)