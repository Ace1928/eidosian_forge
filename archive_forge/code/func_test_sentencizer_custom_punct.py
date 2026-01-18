import pytest
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
@pytest.mark.parametrize('punct_chars,words,sent_starts,sent_ends,n_sents', [(['~', '?'], ['Hello', 'world', '~', 'A', '.', 'B', '.'], [True, False, False, True, False, False, False], [False, False, True, False, False, False, True], 2), (['.', 'ö'], ['Hello', '.', 'Test', 'ö', 'Ok', '.'], [True, False, True, False, True, False], [False, True, False, True, False, True], 3)])
def test_sentencizer_custom_punct(en_vocab, punct_chars, words, sent_starts, sent_ends, n_sents):
    doc = Doc(en_vocab, words=words)
    sentencizer = Sentencizer(punct_chars=punct_chars)
    doc = sentencizer(doc)
    assert doc.has_annotation('SENT_START')
    assert [t.is_sent_start for t in doc] == sent_starts
    assert [t.is_sent_end for t in doc] == sent_ends
    assert len(list(doc.sents)) == n_sents