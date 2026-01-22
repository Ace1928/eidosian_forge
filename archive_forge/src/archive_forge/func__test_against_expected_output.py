import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def _test_against_expected_output(self, stemmer_mode, expected_stems):
    stemmer = PorterStemmer(mode=stemmer_mode)
    for word, true_stem in zip(self._vocabulary(), expected_stems):
        our_stem = stemmer.stem(word)
        assert our_stem == true_stem, '{} should stem to {} in {} mode but got {}'.format(word, true_stem, stemmer_mode, our_stem)