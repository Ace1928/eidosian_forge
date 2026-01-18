import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_spanish(self):
    stemmer = SnowballStemmer('spanish')
    assert stemmer.stem('Visionado') == 'vision'
    assert stemmer.stem('algue') == 'algu'