import pytest
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
def test_sentencizer(en_vocab):
    doc = Doc(en_vocab, words=['Hello', '!', 'This', 'is', 'a', 'test', '.'])
    sentencizer = Sentencizer(punct_chars=None)
    doc = sentencizer(doc)
    assert doc.has_annotation('SENT_START')
    sent_starts = [t.is_sent_start for t in doc]
    sent_ends = [t.is_sent_end for t in doc]
    assert sent_starts == [True, False, True, False, False, False, False]
    assert sent_ends == [False, True, False, False, False, False, True]
    assert len(list(doc.sents)) == 2