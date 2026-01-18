import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_vocabulary_nltk_mode(self):
    with closing(data.find('stemmers/porter_test/porter_nltk_output.txt').open(encoding='utf-8')) as fp:
        self._test_against_expected_output(PorterStemmer.NLTK_EXTENSIONS, fp.read().splitlines())