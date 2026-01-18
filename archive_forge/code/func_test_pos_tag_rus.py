import unittest
from nltk import pos_tag, word_tokenize
def test_pos_tag_rus(self):
    text = 'Илья оторопел и дважды перечитал бумажку.'
    expected_tagged = [('Илья', 'S'), ('оторопел', 'V'), ('и', 'CONJ'), ('дважды', 'ADV'), ('перечитал', 'V'), ('бумажку', 'S'), ('.', 'NONLEX')]
    assert pos_tag(word_tokenize(text), lang='rus') == expected_tagged