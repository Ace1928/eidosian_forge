import pytest
from spacy.lang.ta import Tamil
from spacy.symbols import ORTH
@pytest.mark.parametrize('text,expected_tokens', [('ஆப்பிள் நிறுவனம் யு.கே. தொடக்க நிறுவனத்தை ஒரு லட்சம் கோடிக்கு வாங்கப் பார்க்கிறது', ['ஆப்பிள்', 'நிறுவனம்', 'யு.கே.', 'தொடக்க', 'நிறுவனத்தை', 'ஒரு', 'லட்சம்', 'கோடிக்கு', 'வாங்கப்', 'பார்க்கிறது'])])
def test_ta_tokenizer_special_case(text, expected_tokens):
    nlp = Tamil()
    nlp.tokenizer.add_special_case('யு.கே.', [{ORTH: 'யு.கே.'}])
    tokens = nlp(text)
    token_list = [token.text for token in tokens]
    assert expected_tokens == token_list