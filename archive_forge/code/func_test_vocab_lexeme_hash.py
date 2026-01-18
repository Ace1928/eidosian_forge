import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT
from spacy.lookups import Lookups
from spacy.tokens import Doc
from spacy.util import OOV_RANK
from spacy.vocab import Vocab
@pytest.mark.parametrize('text1,text2', [('phantom', 'opera')])
def test_vocab_lexeme_hash(en_vocab, text1, text2):
    """Test that lexemes are hashable."""
    lex1 = en_vocab[text1]
    lex2 = en_vocab[text2]
    lexes = {lex1: lex1, lex2: lex2}
    assert lexes[lex1].orth_ == text1
    assert lexes[lex2].orth_ == text2