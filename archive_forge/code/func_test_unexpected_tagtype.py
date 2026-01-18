from unittest import TestCase
from unittest.mock import MagicMock
import pytest
from nltk.parse import corenlp
from nltk.tree import Tree
def test_unexpected_tagtype(self):
    with self.assertRaises(ValueError):
        corenlp_tagger = corenlp.CoreNLPParser(tagtype='test')