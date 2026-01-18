import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_words(self):
    self.assertEqual(ptb.words('WSJ/00/WSJ_0003.MRG')[:7], ['A', 'form', 'of', 'asbestos', 'once', 'used', '*'])