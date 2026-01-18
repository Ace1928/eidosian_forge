import unittest
from os import environ, path, sep
from nltk.classify import Senna
from nltk.tag import SennaChunkTagger, SennaNERTagger, SennaTagger
def test_senna_tagger(self):
    tagger = SennaTagger(SENNA_EXECUTABLE_PATH)
    result = tagger.tag('What is the airspeed of an unladen swallow ?'.split())
    expected = [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'), ('unladen', 'NN'), ('swallow', 'NN'), ('?', '.')]
    self.assertEqual(result, expected)