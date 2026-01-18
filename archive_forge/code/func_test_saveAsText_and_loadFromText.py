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
def test_saveAsText_and_loadFromText(self):
    """`Dictionary` can be saved as textfile and loaded again from textfile. """
    tmpf = get_tmpfile('dict_test.txt')
    for sort_by_word in [True, False]:
        d = Dictionary(self.texts)
        d.save_as_text(tmpf, sort_by_word=sort_by_word)
        self.assertTrue(os.path.exists(tmpf))
        d_loaded = Dictionary.load_from_text(tmpf)
        self.assertNotEqual(d_loaded, None)
        self.assertEqual(d_loaded.token2id, d.token2id)