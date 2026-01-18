import unittest
from nltk.corpus import brown
from nltk.jsontags import JSONTaggedDecoder, JSONTaggedEncoder
from nltk.tag import (
from nltk.tag.brill import nltkdemo18
def test_regexp_tagger(self):
    tagger = RegexpTagger([('.*', 'NN')], backoff=self.default_tagger)
    encoded = self.encoder.encode(tagger)
    decoded = self.decoder.decode(encoded)
    self.assertEqual(repr(tagger), repr(decoded))
    self.assertEqual(repr(tagger.backoff), repr(decoded.backoff))
    self.assertEqual(tagger._regexps, decoded._regexps)