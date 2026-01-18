import pytest
from thinc.api import ConfigValidationError
from spacy.lang.zh import Chinese, _get_pkuseg_trie_data
def test_zh_tokenizer_pkuseg_user_dict(zh_tokenizer_pkuseg, zh_tokenizer_char):
    user_dict = _get_pkuseg_trie_data(zh_tokenizer_pkuseg.pkuseg_seg.preprocesser.trie)
    zh_tokenizer_pkuseg.pkuseg_update_user_dict(['nonsense_asdf'])
    updated_user_dict = _get_pkuseg_trie_data(zh_tokenizer_pkuseg.pkuseg_seg.preprocesser.trie)
    assert len(user_dict) == len(updated_user_dict) - 1
    zh_tokenizer_pkuseg.pkuseg_update_user_dict([], reset=True)
    reset_user_dict = _get_pkuseg_trie_data(zh_tokenizer_pkuseg.pkuseg_seg.preprocesser.trie)
    assert len(reset_user_dict) == 0
    with pytest.warns(UserWarning):
        zh_tokenizer_char.pkuseg_update_user_dict(['nonsense_asdf'])