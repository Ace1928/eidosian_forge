from collections import namedtuple
import unittest
import logging
import numpy as np
import pytest
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
class TestTranslationMatrix(unittest.TestCase):

    def setUp(self):
        self.source_word_vec_file = datapath('EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt')
        self.target_word_vec_file = datapath('IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt')
        self.word_pairs = [('one', 'uno'), ('two', 'due'), ('three', 'tre'), ('four', 'quattro'), ('five', 'cinque'), ('seven', 'sette'), ('eight', 'otto'), ('dog', 'cane'), ('pig', 'maiale'), ('fish', 'cavallo'), ('birds', 'uccelli'), ('apple', 'mela'), ('orange', 'arancione'), ('grape', 'acino'), ('banana', 'banana')]
        self.test_word_pairs = [('ten', 'dieci'), ('cat', 'gatto')]
        self.source_word_vec = KeyedVectors.load_word2vec_format(self.source_word_vec_file, binary=False)
        self.target_word_vec = KeyedVectors.load_word2vec_format(self.target_word_vec_file, binary=False)

    def test_translation_matrix(self):
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        self.assertEqual(model.translation_matrix.shape, (300, 300))

    def test_persistence(self):
        """Test storing/loading the entire model."""
        tmpf = get_tmpfile('transmat-en-it.pkl')
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        model.save(tmpf)
        loaded_model = translation_matrix.TranslationMatrix.load(tmpf)
        self.assertTrue(np.allclose(model.translation_matrix, loaded_model.translation_matrix))

    def test_translate_nn(self):
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        test_source_word, test_target_word = zip(*self.test_word_pairs)
        translated_words = model.translate(test_source_word, topn=5, source_lang_vec=self.source_word_vec, target_lang_vec=self.target_word_vec)
        for idx, item in enumerate(self.test_word_pairs):
            self.assertTrue(item[1] in translated_words[item[0]])

    @pytest.mark.xfail(True, reason='blinking test, can be related to <https://github.com/RaRe-Technologies/gensim/issues/2977>')
    def test_translate_gc(self):
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        test_source_word, test_target_word = zip(*self.test_word_pairs)
        translated_words = model.translate(test_source_word, topn=5, gc=1, sample_num=3, source_lang_vec=self.source_word_vec, target_lang_vec=self.target_word_vec)
        for idx, item in enumerate(self.test_word_pairs):
            self.assertTrue(item[1] in translated_words[item[0]])