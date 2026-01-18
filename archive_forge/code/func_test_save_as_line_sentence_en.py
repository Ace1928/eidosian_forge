import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_save_as_line_sentence_en(self):
    corpus_file = get_tmpfile('gensim_utils.tst')
    ref_sentences = [line.split() for line in utils.any2unicode('hello world\nhow are you').split('\n')]
    utils.save_as_line_sentence(ref_sentences, corpus_file)
    with utils.open(corpus_file, 'rb', encoding='utf8') as fin:
        sentences = [line.strip().split() for line in fin.read().strip().split('\n')]
        self.assertEqual(sentences, ref_sentences)