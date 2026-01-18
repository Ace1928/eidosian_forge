import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_retrieve_synsets(self):
    self.assertEqual(sorted(wn.synsets('zap', pos='n')), [S('zap.n.01')])
    self.assertEqual(sorted(wn.synsets('zap', pos='v')), [S('microwave.v.01'), S('nuke.v.01'), S('zap.v.01'), S('zap.v.02')])