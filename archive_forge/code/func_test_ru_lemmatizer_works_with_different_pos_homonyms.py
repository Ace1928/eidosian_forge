import pytest
from spacy.tokens import Doc
@pytest.mark.parametrize('text,pos,morph,lemma', [('рой', 'NOUN', '', 'рой'), ('рой', 'VERB', '', 'рыть'), ('клей', 'NOUN', '', 'клей'), ('клей', 'VERB', '', 'клеить'), ('три', 'NUM', '', 'три'), ('кос', 'NOUN', 'Number=Sing', 'кос'), ('кос', 'NOUN', 'Number=Plur', 'коса'), ('кос', 'ADJ', '', 'косой'), ('потом', 'NOUN', '', 'пот'), ('потом', 'ADV', '', 'потом')])
def test_ru_lemmatizer_works_with_different_pos_homonyms(ru_lemmatizer, text, pos, morph, lemma):
    doc = Doc(ru_lemmatizer.vocab, words=[text], pos=[pos], morphs=[morph])
    result_lemmas = ru_lemmatizer.pymorphy2_lemmatize(doc[0])
    assert result_lemmas == [lemma]