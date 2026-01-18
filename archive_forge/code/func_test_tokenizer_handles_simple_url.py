import pytest
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
@pytest.mark.parametrize('url', URLS_BASIC)
def test_tokenizer_handles_simple_url(tokenizer, url):
    tokens = tokenizer(url)
    assert len(tokens) == 1
    assert tokens[0].text == url