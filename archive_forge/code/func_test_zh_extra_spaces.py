import pytest
from thinc.api import ConfigValidationError
from spacy.lang.zh import Chinese, _get_pkuseg_trie_data
def test_zh_extra_spaces(zh_tokenizer_char):
    tokens = zh_tokenizer_char('I   like cheese.')
    assert tokens[1].orth_ == '  '