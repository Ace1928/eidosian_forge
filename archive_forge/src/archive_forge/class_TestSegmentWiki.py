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
class TestSegmentWiki(unittest.TestCase):

    def setUp(self):
        self.fname = datapath('enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2')
        self.expected_title = 'Anarchism'
        self.expected_section_titles = ['Introduction', 'Etymology and terminology', 'History', 'Anarchist schools of thought', 'Internal issues and debates', 'Topics of interest', 'Criticisms', 'References', 'Further reading', 'External links']

    def tearDown(self):
        fname = get_tmpfile('script.tst')
        extensions = ['', '.json']
        for ext in extensions:
            try:
                os.remove(fname + ext)
            except OSError:
                pass

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

    def test_generator_len(self):
        expected_num_articles = 106
        num_articles = sum((1 for x in segment_all_articles(self.fname)))
        self.assertEqual(num_articles, expected_num_articles)

    def test_json_len(self):
        tmpf = get_tmpfile('script.tst.json')
        segment_and_write_all_articles(self.fname, tmpf, workers=1)
        expected_num_articles = 106
        with utils.open(tmpf, 'rb') as f:
            num_articles = sum((1 for line in f))
        self.assertEqual(num_articles, expected_num_articles)

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