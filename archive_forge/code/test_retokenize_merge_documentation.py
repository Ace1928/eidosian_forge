import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
Test that the retokenizer automatically skips duplicate spans instead
    of complaining about overlaps. See #3687.