import pytest
from spacy.lang.zh import Chinese
from ...util import make_tempdir
def test_zh_tokenizer_serialize_jieba(zh_tokenizer_jieba):
    zh_tokenizer_serialize(zh_tokenizer_jieba)