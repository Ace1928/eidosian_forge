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
def test_segment_all_articles(self):
    title, sections, interlinks = next(segment_all_articles(self.fname, include_interlinks=True))
    self.assertEqual(title, self.expected_title)
    section_titles = [s[0] for s in sections]
    self.assertEqual(section_titles, self.expected_section_titles)
    first_section_text = sections[0][1]
    first_sentence = "'''Anarchism''' is a political philosophy that advocates self-governed societies"
    self.assertTrue(first_sentence in first_section_text)
    self.assertEqual(len(interlinks), 685)
    self.assertTrue(interlinks[0] == ('political philosophy', 'political philosophy'))
    self.assertTrue(interlinks[1] == ('self-governance', 'self-governed'))
    self.assertTrue(interlinks[2] == ('stateless society', 'stateless societies'))