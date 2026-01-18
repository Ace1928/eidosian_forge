import pytest
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
@pytest.mark.parametrize('url', URLS_SHOULD_NOT_MATCH)
def test_should_not_match(en_tokenizer, url):
    assert en_tokenizer.url_match(url) is None