import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_strip_stopword_tokens(self):
    self.assertEqual(remove_stopword_tokens(['the', 'world', 'is', 'sphere']), ['world', 'sphere'])
    with mock.patch('gensim.parsing.preprocessing.STOPWORDS', frozenset(['the'])):
        self.assertEqual(remove_stopword_tokens(['the', 'world', 'is', 'sphere']), ['world', 'is', 'sphere'])