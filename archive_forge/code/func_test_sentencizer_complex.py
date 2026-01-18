import pytest
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
@pytest.mark.parametrize('words,sent_starts,sent_ends,n_sents', [(['Hello', '!', '.', 'Test', '.', '.', 'ok'], [True, False, False, True, False, False, True], [False, False, True, False, False, True, True], 3), (['¡', 'Buen', 'día', '!', 'Hola', ',', '¿', 'qué', 'tal', '?'], [True, False, False, False, True, False, False, False, False, False], [False, False, False, True, False, False, False, False, False, True], 2), (['"', 'Nice', '!', '"', 'I', 'am', 'happy', '.'], [True, False, False, False, True, False, False, False], [False, False, False, True, False, False, False, True], 2)])
def test_sentencizer_complex(en_vocab, words, sent_starts, sent_ends, n_sents):
    doc = Doc(en_vocab, words=words)
    sentencizer = Sentencizer(punct_chars=None)
    doc = sentencizer(doc)
    assert doc.has_annotation('SENT_START')
    assert [t.is_sent_start for t in doc] == sent_starts
    assert [t.is_sent_end for t in doc] == sent_ends
    assert len(list(doc.sents)) == n_sents