import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text,punct', [("(sa'yo", '(')])
def test_tl_tokenizer_splits_pre_punct_regex(text, punct):
    tl_search_prefixes = compile_prefix_regex(TOKENIZER_PREFIXES).search
    match = tl_search_prefixes(text)
    assert match.group() == punct