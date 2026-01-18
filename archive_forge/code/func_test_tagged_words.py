import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_tagged_words(self):
    self.assertEqual(ptb.tagged_words('WSJ/00/WSJ_0003.MRG')[:3], [('A', 'DT'), ('form', 'NN'), ('of', 'IN')])