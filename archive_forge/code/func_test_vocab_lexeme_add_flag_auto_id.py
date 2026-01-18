import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT
from spacy.lookups import Lookups
from spacy.tokens import Doc
from spacy.util import OOV_RANK
from spacy.vocab import Vocab
def test_vocab_lexeme_add_flag_auto_id(en_vocab):
    is_len4 = en_vocab.add_flag(lambda string: len(string) == 4)
    assert en_vocab['1999'].check_flag(is_len4) is True
    assert en_vocab['1999'].check_flag(IS_DIGIT) is True
    assert en_vocab['199'].check_flag(is_len4) is False
    assert en_vocab['199'].check_flag(IS_DIGIT) is True
    assert en_vocab['the'].check_flag(is_len4) is False
    assert en_vocab['dogs'].check_flag(is_len4) is True