import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_retrieve_synset(self):
    move_synset = S('go.v.21')
    self.assertEqual(move_synset.name(), 'move.v.15')
    self.assertEqual(move_synset.lemma_names(), ['move', 'go'])
    self.assertEqual(move_synset.definition(), "have a turn; make one's move in a game")
    self.assertEqual(move_synset.examples(), ['Can I go now?'])