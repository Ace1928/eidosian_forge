import logging
import unittest
import os
import zlib
from gensim.corpora.hashdictionary import HashDictionary
from gensim.test.utils import get_tmpfile, common_texts
def test_saveAsTextBz2(self):
    """ `HashDictionary` can be saved & loaded as compressed pickle. """
    tmpf = get_tmpfile('dict_test.txt.bz2')
    d = HashDictionary(['žloťoučký koníček'.split(), 'Малйж обльйквюэ ат эжт'.split()])
    d.save(tmpf)
    self.assertTrue(os.path.exists(tmpf))
    d2 = d.load(tmpf)
    self.assertEqual(len(d), len(d2))