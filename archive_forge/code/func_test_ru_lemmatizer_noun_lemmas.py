import pytest
from spacy.tokens import Doc
@pytest.mark.parametrize('text,lemmas', [('гвоздики', ['гвоздик', 'гвоздика']), ('люди', ['человек']), ('реки', ['река']), ('кольцо', ['кольцо']), ('пепперони', ['пепперони'])])
def test_ru_lemmatizer_noun_lemmas(ru_lemmatizer, text, lemmas):
    doc = Doc(ru_lemmatizer.vocab, words=[text], pos=['NOUN'])
    result_lemmas = ru_lemmatizer.pymorphy2_lemmatize(doc[0])
    assert sorted(result_lemmas) == lemmas