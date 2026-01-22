from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
class TestWikiCorpus(TestTextCorpus):

    def setUp(self):
        self.corpus_class = wikicorpus.WikiCorpus
        self.file_extension = '.xml.bz2'
        self.fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        self.enwiki = datapath('enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2')

    def test_default_preprocessing(self):
        expected = ['computer', 'human', 'interface']
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        first_text = next(corpus.get_texts())
        self.assertEqual(expected, first_text)

    def test_len(self):
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        all_articles = corpus.get_texts()
        assert len(list(all_articles)) == 9
        corpus = self.corpus_class(self.fname, article_min_tokens=100000)
        all_articles = corpus.get_texts()
        assert len(list(all_articles)) == 0

    def test_load_with_metadata(self):
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        corpus.metadata = True
        self.assertEqual(len(corpus), 9)
        docs = list(corpus)
        self.assertEqual(len(docs), 9)
        for i, docmeta in enumerate(docs):
            doc, metadata = docmeta
            article_no = i + 1
            self.assertEqual(metadata[0], str(article_no))
            self.assertEqual(metadata[1], 'Article%d' % article_no)

    def test_load(self):
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        docs = list(corpus)
        self.assertEqual(len(docs), 9)

    def test_first_element(self):
        """
        First two articles in this sample are
        1) anarchism
        2) autism
        """
        corpus = self.corpus_class(self.enwiki, processes=1)
        texts = corpus.get_texts()
        self.assertTrue(u'anarchism' in next(texts))
        self.assertTrue(u'autism' in next(texts))

    def test_unicode_element(self):
        """
        First unicode article in this sample is
        1) папа
        """
        bgwiki = datapath('bgwiki-latest-pages-articles-shortened.xml.bz2')
        corpus = self.corpus_class(bgwiki)
        texts = corpus.get_texts()
        self.assertTrue(u'папа' in next(texts))

    def test_custom_tokenizer(self):
        """
        define a custom tokenizer function and use it
        """
        wc = self.corpus_class(self.enwiki, processes=1, tokenizer_func=custom_tokenizer, token_max_len=16, token_min_len=1, lower=False)
        row = wc.get_texts()
        list_tokens = next(row)
        self.assertTrue(u'Anarchism' in list_tokens)
        self.assertTrue(u'collectivization' in list_tokens)
        self.assertTrue(u'a' in list_tokens)
        self.assertTrue(u'i.e.' in list_tokens)

    def test_lower_case_set_true(self):
        """
        Set the parameter lower to True and check that upper case 'Anarchism' token doesnt exist
        """
        corpus = self.corpus_class(self.enwiki, processes=1, lower=True)
        row = corpus.get_texts()
        list_tokens = next(row)
        self.assertTrue(u'Anarchism' not in list_tokens)
        self.assertTrue(u'anarchism' in list_tokens)

    def test_lower_case_set_false(self):
        """
        Set the parameter lower to False and check that upper case Anarchism' token exists
        """
        corpus = self.corpus_class(self.enwiki, processes=1, lower=False)
        row = corpus.get_texts()
        list_tokens = next(row)
        self.assertTrue(u'Anarchism' in list_tokens)
        self.assertTrue(u'anarchism' in list_tokens)

    def test_min_token_len_not_set(self):
        """
        Don't set the parameter token_min_len and check that 'a' as a token doesn't exist
        Default token_min_len=2
        """
        corpus = self.corpus_class(self.enwiki, processes=1)
        self.assertTrue(u'a' not in next(corpus.get_texts()))

    def test_min_token_len_set(self):
        """
        Set the parameter token_min_len to 1 and check that 'a' as a token exists
        """
        corpus = self.corpus_class(self.enwiki, processes=1, token_min_len=1)
        self.assertTrue(u'a' in next(corpus.get_texts()))

    def test_max_token_len_not_set(self):
        """
        Don't set the parameter token_max_len and check that 'collectivisation' as a token doesn't exist
        Default token_max_len=15
        """
        corpus = self.corpus_class(self.enwiki, processes=1)
        self.assertTrue(u'collectivization' not in next(corpus.get_texts()))

    def test_max_token_len_set(self):
        """
        Set the parameter token_max_len to 16 and check that 'collectivisation' as a token exists
        """
        corpus = self.corpus_class(self.enwiki, processes=1, token_max_len=16)
        self.assertTrue(u'collectivization' in next(corpus.get_texts()))

    def test_removed_table_markup(self):
        """
        Check if all the table markup has been removed.
        """
        enwiki_file = datapath('enwiki-table-markup.xml.bz2')
        corpus = self.corpus_class(enwiki_file)
        texts = corpus.get_texts()
        table_markup = ['style', 'class', 'border', 'cellspacing', 'cellpadding', 'colspan', 'rowspan']
        for text in texts:
            for word in table_markup:
                self.assertTrue(word not in text)

    def test_get_stream(self):
        wiki = self.corpus_class(self.enwiki)
        sample_text_wiki = next(wiki.getstream()).decode()[1:14]
        self.assertEqual(sample_text_wiki, 'mediawiki xml')

    def test_sample_text(self):
        pass

    def test_sample_text_length(self):
        pass

    def test_sample_text_seed(self):
        pass

    def test_empty_input(self):
        pass

    def test_custom_filterfunction(self):

        def reject_all(elem, *args, **kwargs):
            return False
        corpus = self.corpus_class(self.enwiki, filter_articles=reject_all)
        texts = corpus.get_texts()
        self.assertFalse(any(texts))

        def keep_some(elem, title, *args, **kwargs):
            return title[0] == 'C'
        corpus = self.corpus_class(self.enwiki, filter_articles=reject_all)
        corpus.metadata = True
        texts = corpus.get_texts()
        for text, (pageid, title) in texts:
            self.assertEquals(title[0], 'C')