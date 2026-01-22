import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
class TestSinicaTreebank(unittest.TestCase):

    def test_sents(self):
        first_3_sents = sinica_treebank.sents()[:3]
        self.assertEqual(first_3_sents, [['一'], ['友情'], ['嘉珍', '和', '我', '住在', '同一條', '巷子']])

    def test_parsed_sents(self):
        parsed_sents = sinica_treebank.parsed_sents()[25]
        self.assertEqual(parsed_sents, Tree('S', [Tree('NP', [Tree('Nba', ['嘉珍'])]), Tree('V‧地', [Tree('VA11', ['不停']), Tree('DE', ['的'])]), Tree('VA4', ['哭泣'])]))