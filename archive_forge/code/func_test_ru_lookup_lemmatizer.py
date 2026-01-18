import pytest
from spacy.tokens import Doc
@pytest.mark.parametrize('word,lemma', (('бременем', 'бремя'), ('будешь', 'быть'), ('какая-то', 'какой-то')))
def test_ru_lookup_lemmatizer(ru_lookup_lemmatizer, word, lemma):
    assert ru_lookup_lemmatizer.mode == 'pymorphy3_lookup'
    doc = Doc(ru_lookup_lemmatizer.vocab, words=[word])
    assert ru_lookup_lemmatizer(doc)[0].lemma_ == lemma