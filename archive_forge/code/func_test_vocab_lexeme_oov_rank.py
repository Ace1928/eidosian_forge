import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT
from spacy.lookups import Lookups
from spacy.tokens import Doc
from spacy.util import OOV_RANK
from spacy.vocab import Vocab
def test_vocab_lexeme_oov_rank(en_vocab):
    """Test that default rank is OOV_RANK."""
    lex = en_vocab['word']
    assert OOV_RANK == numpy.iinfo(numpy.uint64).max
    assert lex.rank == OOV_RANK