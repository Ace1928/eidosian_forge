from collections.abc import Mapping
from itertools import chain
import logging
import unittest
import codecs
import os
import os.path
import scipy
import gensim
from gensim.corpora import Dictionary
from gensim.utils import to_utf8
from gensim.test.utils import get_tmpfile, common_texts
def test_saveAsText(self):
    """`Dictionary` can be saved as textfile. """
    tmpf = get_tmpfile('save_dict_test.txt')
    small_text = [['prvé', 'slovo'], ['slovo', 'druhé'], ['druhé', 'slovo']]
    d = Dictionary(small_text)
    d.save_as_text(tmpf)
    with codecs.open(tmpf, 'r', encoding='utf-8') as file:
        serialized_lines = file.readlines()
        self.assertEqual(serialized_lines[0], u'3\n')
        self.assertEqual(len(serialized_lines), 4)
        self.assertEqual(serialized_lines[1][1:], u'\tdruhé\t2\n')
        self.assertEqual(serialized_lines[2][1:], u'\tprvé\t1\n')
        self.assertEqual(serialized_lines[3][1:], u'\tslovo\t3\n')
    d.save_as_text(tmpf, sort_by_word=False)
    with codecs.open(tmpf, 'r', encoding='utf-8') as file:
        serialized_lines = file.readlines()
        self.assertEqual(serialized_lines[0], u'3\n')
        self.assertEqual(len(serialized_lines), 4)
        self.assertEqual(serialized_lines[1][1:], u'\tslovo\t3\n')
        self.assertEqual(serialized_lines[2][1:], u'\tdruhé\t2\n')
        self.assertEqual(serialized_lines[3][1:], u'\tprvé\t1\n')