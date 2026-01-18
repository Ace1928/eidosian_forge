import unittest
from nltk import pos_tag, word_tokenize
def test_pos_tag_rus_universal(self):
    text = 'Илья оторопел и дважды перечитал бумажку.'
    expected_tagged = [('Илья', 'NOUN'), ('оторопел', 'VERB'), ('и', 'CONJ'), ('дважды', 'ADV'), ('перечитал', 'VERB'), ('бумажку', 'NOUN'), ('.', '.')]
    assert pos_tag(word_tokenize(text), tagset='universal', lang='rus') == expected_tagged