import pytest
@pytest.mark.xfail
def test_ordinal_number(sk_tokenizer):
    text = '10. decembra 1948'
    tokens = sk_tokenizer(text)
    assert len(tokens) == 3