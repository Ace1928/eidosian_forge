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
def test_patch_with_special_tokens(self):
    special_tokens = {'pad': 0, 'space': 1, 'quake': 3}
    corpus = [['máma', 'mele', 'maso'], ['ema', 'má', 'máma']]
    d = Dictionary(corpus)
    self.assertEqual(len(d.token2id), 5)
    d.patch_with_special_tokens(special_tokens)
    self.assertEqual(d.token2id['pad'], 0)
    self.assertEqual(d.token2id['space'], 1)
    self.assertEqual(d.token2id['quake'], 3)
    self.assertEqual(len(d.token2id), 8)
    self.assertNotIn((0, 1), d.doc2bow(corpus[0]))
    self.assertIn((0, 1), d.doc2bow(['pad'] + corpus[0]))
    corpus_with_special_tokens = [['máma', 'mele', 'maso'], ['ema', 'má', 'máma', 'space']]
    d = Dictionary(corpus_with_special_tokens)
    self.assertEqual(len(d.token2id), 6)
    self.assertNotEqual(d.token2id['space'], 1)
    d.patch_with_special_tokens(special_tokens)
    self.assertEqual(len(d.token2id), 8)
    self.assertEqual(max(d.token2id.values()), 7)
    self.assertEqual(d.token2id['space'], 1)
    self.assertNotIn((1, 1), d.doc2bow(corpus_with_special_tokens[0]))
    self.assertIn((1, 1), d.doc2bow(corpus_with_special_tokens[1]))