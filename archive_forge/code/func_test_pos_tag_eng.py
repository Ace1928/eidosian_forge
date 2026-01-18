import unittest
from nltk import pos_tag, word_tokenize
def test_pos_tag_eng(self):
    text = "John's big idea isn't all that bad."
    expected_tagged = [('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ'), ("n't", 'RB'), ('all', 'PDT'), ('that', 'DT'), ('bad', 'JJ'), ('.', '.')]
    assert pos_tag(word_tokenize(text)) == expected_tagged