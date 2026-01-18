import pytest
from spacy.tokens import Doc
def test_ru_doc_lemmatization(ru_lemmatizer):
    words = ['мама', 'мыла', 'раму']
    pos = ['NOUN', 'VERB', 'NOUN']
    morphs = ['Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing', 'Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act', 'Animacy=Anim|Case=Acc|Gender=Fem|Number=Sing']
    doc = Doc(ru_lemmatizer.vocab, words=words, pos=pos, morphs=morphs)
    doc = ru_lemmatizer(doc)
    lemmas = [token.lemma_ for token in doc]
    assert lemmas == ['мама', 'мыть', 'рама']