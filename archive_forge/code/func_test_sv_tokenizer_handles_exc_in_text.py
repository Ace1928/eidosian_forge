import pytest
def test_sv_tokenizer_handles_exc_in_text(sv_tokenizer):
    text = 'Det är bl.a. inte meningen'
    tokens = sv_tokenizer(text)
    assert len(tokens) == 5
    assert tokens[2].text == 'bl.a.'