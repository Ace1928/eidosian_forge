import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_matching_files(self):
    """(wtree) Search for matches or no matches only"""
    tree = self.make_branch_and_tree('.')
    contents = ['d/', 'd/aaa', 'bbb']
    self.build_tree(contents)
    tree.add(contents)
    tree.commit('Initial commit')
    streams = self.run_bzr(['grep', '--color', 'always', '--files-with-matches', 'aaa'])
    self.assertEqual(streams, (''.join([FG.MAGENTA, 'd/aaa', FG.NONE, '\n']), ''))
    streams = self.run_bzr(['grep', '--color', 'always', '--files-without-match', 'aaa'])
    self.assertEqual(streams, (''.join([FG.MAGENTA, 'bbb', FG.NONE, '\n']), ''))