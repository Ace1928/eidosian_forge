import pytest
@pytest.mark.parametrize('text', ['driveru', 'hajaru', 'Serru', 'Fixaru'])
def test_sv_tokenizer_handles_verb_exceptions(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[1].text == 'u'