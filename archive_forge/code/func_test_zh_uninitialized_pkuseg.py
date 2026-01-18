import pytest
from thinc.api import ConfigValidationError
from spacy.lang.zh import Chinese, _get_pkuseg_trie_data
def test_zh_uninitialized_pkuseg():
    config = {'nlp': {'tokenizer': {'segmenter': 'char'}}}
    nlp = Chinese.from_config(config)
    nlp.tokenizer.segmenter = 'pkuseg'
    with pytest.raises(ValueError):
        nlp('test')