import os
import pytest
from spacy.attrs import IS_ALPHA, LEMMA, ORTH
from spacy.lang.en import English
from spacy.parts_of_speech import NOUN, VERB
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_vocab_writing_system(en_vocab):
    assert en_vocab.writing_system['direction'] == 'ltr'
    assert en_vocab.writing_system['has_case'] is True