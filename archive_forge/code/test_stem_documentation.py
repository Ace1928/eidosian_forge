import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
Test for improvement on https://github.com/nltk/nltk/issues/2507

        Ensures that stems are lowercased when `to_lowercase=True`
        