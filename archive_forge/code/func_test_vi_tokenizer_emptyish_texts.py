import pytest
from spacy.lang.vi import Vietnamese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
def test_vi_tokenizer_emptyish_texts(vi_tokenizer):
    doc = vi_tokenizer('')
    assert len(doc) == 0
    doc = vi_tokenizer(' ')
    assert len(doc) == 1
    doc = vi_tokenizer('\n\n\n \t\t \n\n\n')
    assert len(doc) == 1