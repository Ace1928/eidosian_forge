import unittest
from nltk import pos_tag, word_tokenize
def test_pos_tag_eng_universal(self):
    text = "John's big idea isn't all that bad."
    expected_tagged = [('John', 'NOUN'), ("'s", 'PRT'), ('big', 'ADJ'), ('idea', 'NOUN'), ('is', 'VERB'), ("n't", 'ADV'), ('all', 'DET'), ('that', 'DET'), ('bad', 'ADJ'), ('.', '.')]
    assert pos_tag(word_tokenize(text), tagset='universal') == expected_tagged