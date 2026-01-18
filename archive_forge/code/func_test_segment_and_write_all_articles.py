from __future__ import unicode_literals
import json
import logging
import os.path
import unittest
import numpy as np
from gensim import utils
from gensim.scripts.segment_wiki import segment_all_articles, segment_and_write_all_articles
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.word2vec2tensor import word2vec2tensor
from gensim.models import KeyedVectors
def test_segment_and_write_all_articles(self):
    tmpf = get_tmpfile('script.tst.json')
    segment_and_write_all_articles(self.fname, tmpf, workers=1, include_interlinks=True)
    with open(tmpf) as f:
        first = next(f)
    article = json.loads(first)
    title, section_titles, interlinks = (article['title'], article['section_titles'], article['interlinks'])
    self.assertEqual(title, self.expected_title)
    self.assertEqual(section_titles, self.expected_section_titles)
    self.assertEqual(len(interlinks), 685)
    self.assertEqual(tuple(interlinks[0]), ('political philosophy', 'political philosophy'))
    self.assertEqual(tuple(interlinks[1]), ('self-governance', 'self-governed'))
    self.assertEqual(tuple(interlinks[2]), ('stateless society', 'stateless societies'))