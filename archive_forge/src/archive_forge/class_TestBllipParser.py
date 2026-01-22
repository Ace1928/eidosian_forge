import pytest
from nltk.data import find
from nltk.parse.bllip import BllipParser
from nltk.tree import Tree
class TestBllipParser:

    def test_parser_loads_a_valid_tree(self, parser):
        parsed = parser.parse('I saw the man with the telescope')
        tree = next(parsed)
        assert isinstance(tree, Tree)
        assert tree.pformat() == '\n(S1\n  (S\n    (NP (PRP I))\n    (VP\n      (VBD saw)\n      (NP (DT the) (NN man))\n      (PP (IN with) (NP (DT the) (NN telescope))))))\n'.strip()

    def test_tagged_parse_finds_matching_element(self, parser):
        parsed = parser.parse('I saw the man with the telescope')
        tagged_tree = next(parser.tagged_parse([('telescope', 'NN')]))
        assert isinstance(tagged_tree, Tree)
        assert tagged_tree.pformat() == '(S1 (NP (NN telescope)))'