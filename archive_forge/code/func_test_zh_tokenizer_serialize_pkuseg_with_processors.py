import pytest
from spacy.lang.zh import Chinese
from ...util import make_tempdir
@pytest.mark.slow
def test_zh_tokenizer_serialize_pkuseg_with_processors(zh_tokenizer_pkuseg):
    config = {'nlp': {'tokenizer': {'@tokenizers': 'spacy.zh.ChineseTokenizer', 'segmenter': 'pkuseg'}}, 'initialize': {'tokenizer': {'pkuseg_model': 'medicine'}}}
    nlp = Chinese.from_config(config)
    nlp.initialize()
    zh_tokenizer_serialize(nlp.tokenizer)