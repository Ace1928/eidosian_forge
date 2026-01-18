import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_wordnet_similarities(self):
    self.assertAlmostEqual(S('cat.n.01').path_similarity(S('cat.n.01')), 1.0)
    self.assertAlmostEqual(S('dog.n.01').path_similarity(S('cat.n.01')), 0.2)
    self.assertAlmostEqual(S('car.n.01').path_similarity(S('automobile.v.01')), S('automobile.v.01').path_similarity(S('car.n.01')))
    self.assertAlmostEqual(S('big.a.01').path_similarity(S('dog.n.01')), S('dog.n.01').path_similarity(S('big.a.01')))
    self.assertAlmostEqual(S('big.a.01').path_similarity(S('long.a.01')), S('long.a.01').path_similarity(S('big.a.01')))
    self.assertAlmostEqual(S('dog.n.01').lch_similarity(S('cat.n.01')), 2.028, places=3)
    self.assertAlmostEqual(S('dog.n.01').wup_similarity(S('cat.n.01')), 0.8571, places=3)
    self.assertAlmostEqual(S('car.n.01').wup_similarity(S('automobile.v.01')), S('automobile.v.01').wup_similarity(S('car.n.01')))
    self.assertAlmostEqual(S('big.a.01').wup_similarity(S('dog.n.01')), S('dog.n.01').wup_similarity(S('big.a.01')))
    self.assertAlmostEqual(S('big.a.01').wup_similarity(S('long.a.01')), S('long.a.01').wup_similarity(S('big.a.01')))
    self.assertAlmostEqual(S('big.a.01').lch_similarity(S('long.a.01')), S('long.a.01').lch_similarity(S('big.a.01')))
    brown_ic = wnic.ic('ic-brown.dat')
    self.assertAlmostEqual(S('dog.n.01').jcn_similarity(S('cat.n.01'), brown_ic), 0.4497, places=3)
    semcor_ic = wnic.ic('ic-semcor.dat')
    self.assertAlmostEqual(S('dog.n.01').lin_similarity(S('cat.n.01'), semcor_ic), 0.8863, places=3)