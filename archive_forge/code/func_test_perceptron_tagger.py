import unittest
from nltk.corpus import brown
from nltk.jsontags import JSONTaggedDecoder, JSONTaggedEncoder
from nltk.tag import (
from nltk.tag.brill import nltkdemo18
def test_perceptron_tagger(self):
    tagger = PerceptronTagger(load=False)
    tagger.train(self.corpus)
    encoded = self.encoder.encode(tagger)
    decoded = self.decoder.decode(encoded)
    self.assertEqual(tagger.model.weights, decoded.model.weights)
    self.assertEqual(tagger.tagdict, decoded.tagdict)
    self.assertEqual(tagger.classes, decoded.classes)