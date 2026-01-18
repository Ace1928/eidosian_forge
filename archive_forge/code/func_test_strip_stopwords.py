import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_strip_stopwords(self):
    self.assertEqual(remove_stopwords('the world is square'), 'world square')
    with mock.patch('gensim.parsing.preprocessing.STOPWORDS', frozenset(['the'])):
        self.assertEqual(remove_stopwords('the world is square'), 'world is square')