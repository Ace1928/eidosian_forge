import os
import pytest
from spacy.attrs import IS_ALPHA, LEMMA, ORTH
from spacy.lang.en import English
from spacy.parts_of_speech import NOUN, VERB
from spacy.vocab import Vocab
from ..util import make_tempdir
Test Vocab.__contains__ works with int keys.