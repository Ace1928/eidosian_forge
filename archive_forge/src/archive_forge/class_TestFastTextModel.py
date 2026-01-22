from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
class TestFastTextModel(unittest.TestCase):

    def setUp(self):
        self.test_model_file = datapath('lee_fasttext.bin')
        self.test_model = gensim.models.fasttext.load_facebook_model(self.test_model_file)
        self.test_new_model_file = datapath('lee_fasttext_new.bin')

    def test_training(self):
        model = FT_gensim(vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
        model.build_vocab(sentences)
        self.model_sanity(model)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        sims = model.wv.most_similar('graph', topn=10)
        self.assertEqual(model.wv.vectors.shape, (12, 12))
        self.assertEqual(len(model.wv), 12)
        self.assertEqual(model.wv.vectors_vocab.shape[1], 12)
        self.assertEqual(model.wv.vectors_ngrams.shape[1], 12)
        self.model_sanity(model)
        graph_vector = model.wv.get_vector('graph', norm=True)
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']
        self.assertEqual(sims, sims2)
        model2 = FT_gensim(sentences, vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
        self.models_equal(model, model2)
        invocab_vec = model.wv['minors']
        self.assertEqual(len(invocab_vec), 12)
        oov_vec = model.wv['minor']
        self.assertEqual(len(oov_vec), 12)

    def test_fast_text_train_parameters(self):
        model = FT_gensim(vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
        model.build_vocab(corpus_iterable=sentences)
        self.assertRaises(TypeError, model.train, corpus_file=11111, total_examples=1, epochs=1)
        self.assertRaises(TypeError, model.train, corpus_iterable=11111, total_examples=1, epochs=1)
        self.assertRaises(TypeError, model.train, corpus_iterable=sentences, corpus_file='test', total_examples=1, epochs=1)
        self.assertRaises(TypeError, model.train, corpus_iterable=None, corpus_file=None, total_examples=1, epochs=1)
        self.assertRaises(TypeError, model.train, corpus_file=sentences, total_examples=1, epochs=1)

    def test_training_fromfile(self):
        with temporary_file('gensim_fasttext.tst') as corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            model = FT_gensim(vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
            model.build_vocab(corpus_file=corpus_file)
            self.model_sanity(model)
            model.train(corpus_file=corpus_file, total_words=model.corpus_total_words, epochs=model.epochs)
            sims = model.wv.most_similar('graph', topn=10)
            self.assertEqual(model.wv.vectors.shape, (12, 12))
            self.assertEqual(len(model.wv), 12)
            self.assertEqual(model.wv.vectors_vocab.shape[1], 12)
            self.assertEqual(model.wv.vectors_ngrams.shape[1], 12)
            self.model_sanity(model)
            graph_vector = model.wv.get_vector('graph', norm=True)
            sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
            sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']
            self.assertEqual(sims, sims2)
            invocab_vec = model.wv['minors']
            self.assertEqual(len(invocab_vec), 12)
            oov_vec = model.wv['minor']
            self.assertEqual(len(oov_vec), 12)

    def models_equal(self, model, model2):
        self.assertEqual(len(model.wv), len(model2.wv))
        self.assertEqual(model.wv.bucket, model2.wv.bucket)
        self.assertTrue(np.allclose(model.wv.vectors_vocab, model2.wv.vectors_vocab))
        self.assertTrue(np.allclose(model.wv.vectors_ngrams, model2.wv.vectors_ngrams))
        self.assertTrue(np.allclose(model.wv.vectors, model2.wv.vectors))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        most_common_word = max(model.wv.key_to_index, key=lambda word: model.wv.get_vecattr(word, 'count'))[0]
        self.assertTrue(np.allclose(model.wv[most_common_word], model2.wv[most_common_word]))

    def test_persistence(self):
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model = FT_gensim(sentences, min_count=1, bucket=BUCKET)
        model.save(tmpf)
        self.models_equal(model, FT_gensim.load(tmpf))
        wv = model.wv
        wv.save(tmpf)
        loaded_wv = FastTextKeyedVectors.load(tmpf)
        self.assertTrue(np.allclose(wv.vectors_ngrams, loaded_wv.vectors_ngrams))
        self.assertEqual(len(wv), len(loaded_wv))

    def test_persistence_fromfile(self):
        with temporary_file('gensim_fasttext1.tst') as corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            tmpf = get_tmpfile('gensim_fasttext.tst')
            model = FT_gensim(corpus_file=corpus_file, min_count=1, bucket=BUCKET)
            model.save(tmpf)
            self.models_equal(model, FT_gensim.load(tmpf))
            wv = model.wv
            wv.save(tmpf)
            loaded_wv = FastTextKeyedVectors.load(tmpf)
            self.assertTrue(np.allclose(wv.vectors_ngrams, loaded_wv.vectors_ngrams))
            self.assertEqual(len(wv), len(loaded_wv))

    def model_sanity(self, model):
        self.model_structural_sanity(model)

    def model_structural_sanity(self, model):
        """Check a model for basic self-consistency, necessary properties & property
        correspondences, but no semantic tests."""
        self.assertEqual(model.wv.vectors.shape, (len(model.wv), model.vector_size))
        self.assertEqual(model.wv.vectors_vocab.shape, (len(model.wv), model.vector_size))
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model.vector_size))
        self.assertLessEqual(len(model.wv.vectors_ngrams_lockf), len(model.wv.vectors_ngrams))
        self.assertLessEqual(len(model.wv.vectors_vocab_lockf), len(model.wv.index_to_key))
        self.assertTrue(np.isfinite(model.wv.vectors_ngrams).all(), 'NaN in ngrams')
        self.assertTrue(np.isfinite(model.wv.vectors_vocab).all(), 'NaN in vectors_vocab')
        if model.negative:
            self.assertTrue(np.isfinite(model.syn1neg).all(), 'NaN in syn1neg')
        if model.hs:
            self.assertTrue(np.isfinite(model.syn1).all(), 'NaN in syn1neg')

    def test_load_fasttext_format(self):
        try:
            model = gensim.models.fasttext.load_facebook_model(self.test_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_model_file, exc))
        vocab_size, model_size = (1762, 10)
        self.assertEqual(model.wv.vectors.shape, (vocab_size, model_size))
        self.assertEqual(len(model.wv), vocab_size, model_size)
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model_size))
        expected_vec = [-0.57144, -0.0085561, 0.15748, -0.67855, -0.25459, -0.58077, -0.09913, 1.1447, 0.23418, 0.060007]
        actual_vec = model.wv['hundred']
        self.assertTrue(np.allclose(actual_vec, expected_vec, atol=0.0001))
        expected_vec_oov = [-0.21929, -0.53778, -0.22463, -0.41735, 0.71737, -1.59758, -0.24833, 0.62028, 0.53203, 0.77568]
        actual_vec_oov = model.wv['rejection']
        self.assertTrue(np.allclose(actual_vec_oov, expected_vec_oov, atol=0.0001))
        self.assertEqual(model.min_count, 5)
        self.assertEqual(model.window, 5)
        self.assertEqual(model.epochs, 5)
        self.assertEqual(model.negative, 5)
        self.assertEqual(model.sample, 0.0001)
        self.assertEqual(model.wv.bucket, 1000)
        self.assertEqual(model.wv.max_n, 6)
        self.assertEqual(model.wv.min_n, 3)
        self.assertEqual(model.wv.vectors.shape, (len(model.wv), model.vector_size))
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model.vector_size))

    def test_load_fasttext_new_format(self):
        try:
            new_model = gensim.models.fasttext.load_facebook_model(self.test_new_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_new_model_file, exc))
        vocab_size, model_size = (1763, 10)
        self.assertEqual(new_model.wv.vectors.shape, (vocab_size, model_size))
        self.assertEqual(len(new_model.wv), vocab_size, model_size)
        self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.bucket, model_size))
        expected_vec = [-0.025627, -0.11448, 0.18116, -0.96779, 0.2532, -0.93224, 0.3929, 0.12679, -0.19685, -0.13179]
        actual_vec = new_model.wv['hundred']
        self.assertTrue(np.allclose(actual_vec, expected_vec, atol=0.0001))
        expected_vec_oov = [-0.49111, -0.13122, -0.02109, -0.88769, -0.20105, -0.91732, 0.47243, 0.19708, -0.17856, 0.19815]
        actual_vec_oov = new_model.wv['rejection']
        self.assertTrue(np.allclose(actual_vec_oov, expected_vec_oov, atol=0.0001))
        self.assertEqual(new_model.min_count, 5)
        self.assertEqual(new_model.window, 5)
        self.assertEqual(new_model.epochs, 5)
        self.assertEqual(new_model.negative, 5)
        self.assertEqual(new_model.sample, 0.0001)
        self.assertEqual(new_model.wv.bucket, 1000)
        self.assertEqual(new_model.wv.max_n, 6)
        self.assertEqual(new_model.wv.min_n, 3)
        self.assertEqual(new_model.wv.vectors.shape, (len(new_model.wv), new_model.vector_size))
        self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.bucket, new_model.vector_size))

    def test_load_model_supervised(self):
        with self.assertRaises(NotImplementedError):
            gensim.models.fasttext.load_facebook_model(datapath('pang_lee_polarity_fasttext.bin'))

    def test_load_model_with_non_ascii_vocab(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('non_ascii_fasttext.bin'))
        self.assertTrue(u'který' in model.wv)
        try:
            model.wv[u'který']
        except UnicodeDecodeError:
            self.fail('Unable to access vector for utf8 encoded non-ascii word')

    def test_load_model_non_utf8_encoding(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('cp852_fasttext.bin'), encoding='cp852')
        self.assertTrue(u'který' in model.wv)
        try:
            model.wv[u'který']
        except KeyError:
            self.fail('Unable to access vector for cp-852 word')

    def test_oov_similarity(self):
        word = 'someoovword'
        most_similar = self.test_model.wv.most_similar(word)
        top_neighbor, top_similarity = most_similar[0]
        v1 = self.test_model.wv[word]
        v2 = self.test_model.wv[top_neighbor]
        top_similarity_direct = self.test_model.wv.cosine_similarities(v1, v2.reshape(1, -1))[0]
        self.assertAlmostEqual(top_similarity, top_similarity_direct, places=6)

    def test_n_similarity(self):
        self.assertTrue(np.allclose(self.test_model.wv.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.wv.n_similarity(['the'], ['and']), self.test_model.wv.n_similarity(['and'], ['the']))
        self.assertTrue(np.allclose(self.test_model.wv.n_similarity(['night', 'nights'], ['nights', 'night']), 1.0))
        self.assertEqual(self.test_model.wv.n_similarity(['night'], ['nights']), self.test_model.wv.n_similarity(['nights'], ['night']))

    def test_similarity(self):
        self.assertTrue(np.allclose(self.test_model.wv.similarity('the', 'the'), 1.0))
        self.assertEqual(self.test_model.wv.similarity('the', 'and'), self.test_model.wv.similarity('and', 'the'))
        self.assertTrue(np.allclose(self.test_model.wv.similarity('nights', 'nights'), 1.0))
        self.assertEqual(self.test_model.wv.similarity('night', 'nights'), self.test_model.wv.similarity('nights', 'night'))

    def test_most_similar(self):
        self.assertEqual(len(self.test_model.wv.most_similar(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(self.test_model.wv.most_similar('the'), self.test_model.wv.most_similar(positive=['the']))
        self.assertEqual(len(self.test_model.wv.most_similar(['night', 'nights'], topn=5)), 5)
        self.assertEqual(self.test_model.wv.most_similar('nights'), self.test_model.wv.most_similar(positive=['nights']))

    def test_most_similar_cosmul(self):
        self.assertEqual(len(self.test_model.wv.most_similar_cosmul(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(self.test_model.wv.most_similar_cosmul('the'), self.test_model.wv.most_similar_cosmul(positive=['the']))
        self.assertEqual(len(self.test_model.wv.most_similar_cosmul(['night', 'nights'], topn=5)), 5)
        self.assertEqual(self.test_model.wv.most_similar_cosmul('nights'), self.test_model.wv.most_similar_cosmul(positive=['nights']))
        self.assertEqual(self.test_model.wv.most_similar_cosmul('the', 'and'), self.test_model.wv.most_similar_cosmul(positive=['the'], negative=['and']))

    def test_lookup(self):
        self.assertTrue('night' in self.test_model.wv.key_to_index)
        self.assertTrue(np.allclose(self.test_model.wv['night'], self.test_model.wv[['night']]))
        self.assertFalse('nights' in self.test_model.wv.key_to_index)
        self.assertTrue(np.allclose(self.test_model.wv['nights'], self.test_model.wv[['nights']]))

    def test_contains(self):
        self.assertTrue('night' in self.test_model.wv.key_to_index)
        self.assertTrue('night' in self.test_model.wv)
        self.assertFalse(self.test_model.wv.has_index_for('nights'))
        self.assertFalse('nights' in self.test_model.wv.key_to_index)
        self.assertTrue('nights' in self.test_model.wv)

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_wm_distance(self):
        doc = ['night', 'payment']
        oov_doc = ['nights', 'forests', 'payments']
        dist = self.test_model.wv.wmdistance(doc, oov_doc)
        self.assertNotEqual(float('inf'), dist)

    def test_cbow_neg_training(self):
        model_gensim = FT_gensim(vector_size=48, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=5, min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=0.001, min_n=3, max_n=6, sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET)
        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())
        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = [word for word, distance in sims_gensim]
        expected_sims_words = [u'night.', u'night,', u'eight', u'fight', u'month', u'hearings', u'Washington', u'remains', u'overnight', u'running']
        overlaps = set(sims_gensim_words).intersection(expected_sims_words)
        overlap_count = len(overlaps)
        self.assertGreaterEqual(overlap_count, 2, 'only %i overlap in expected %s & actual %s' % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_cbow_neg_training_fromfile(self):
        with temporary_file('gensim_fasttext.tst') as corpus_file:
            model_gensim = FT_gensim(vector_size=48, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=5, min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=0.001, min_n=3, max_n=6, sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET)
            lee_data = LineSentence(datapath('lee_background.cor'))
            utils.save_as_line_sentence(lee_data, corpus_file)
            model_gensim.build_vocab(corpus_file=corpus_file)
            orig0 = np.copy(model_gensim.wv.vectors[0])
            model_gensim.train(corpus_file=corpus_file, total_words=model_gensim.corpus_total_words, epochs=model_gensim.epochs)
            self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())
            sims_gensim = model_gensim.wv.most_similar('night', topn=10)
            sims_gensim_words = [word for word, distance in sims_gensim]
            expected_sims_words = [u'night.', u'night,', u'eight', u'fight', u'month', u'hearings', u'Washington', u'remains', u'overnight', u'running']
            overlaps = set(sims_gensim_words).intersection(expected_sims_words)
            overlap_count = len(overlaps)
            self.assertGreaterEqual(overlap_count, 2, 'only %i overlap in expected %s & actual %s' % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_sg_neg_training(self):
        model_gensim = FT_gensim(vector_size=48, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=0, negative=5, min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=0.001, min_n=3, max_n=6, sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET * 4)
        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())
        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = [word for word, distance in sims_gensim]
        expected_sims_words = [u'night.', u'night,', u'eight', u'overnight', u'overnight.', u'month', u'land', u'firm', u'singles', u'death']
        overlaps = set(sims_gensim_words).intersection(expected_sims_words)
        overlap_count = len(overlaps)
        self.assertGreaterEqual(overlap_count, 2, 'only %i overlap in expected %s & actual %s' % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_sg_neg_training_fromfile(self):
        with temporary_file('gensim_fasttext.tst') as corpus_file:
            model_gensim = FT_gensim(vector_size=48, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=0, negative=5, min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=0.001, min_n=3, max_n=6, sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET * 4)
            lee_data = LineSentence(datapath('lee_background.cor'))
            utils.save_as_line_sentence(lee_data, corpus_file)
            model_gensim.build_vocab(corpus_file=corpus_file)
            orig0 = np.copy(model_gensim.wv.vectors[0])
            model_gensim.train(corpus_file=corpus_file, total_words=model_gensim.corpus_total_words, epochs=model_gensim.epochs)
            self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())
            sims_gensim = model_gensim.wv.most_similar('night', topn=10)
            sims_gensim_words = [word for word, distance in sims_gensim]
            expected_sims_words = [u'night.', u'night,', u'eight', u'overnight', u'overnight.', u'month', u'land', u'firm', u'singles', u'death']
            overlaps = set(sims_gensim_words).intersection(expected_sims_words)
            overlap_count = len(overlaps)
            self.assertGreaterEqual(overlap_count, 2, 'only %i overlap in expected %s & actual %s' % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_online_learning(self):
        model_hs = FT_gensim(sentences, vector_size=12, min_count=1, seed=42, hs=1, negative=0, bucket=BUCKET)
        self.assertEqual(len(model_hs.wv), 12)
        self.assertEqual(model_hs.wv.get_vecattr('graph', 'count'), 3)
        model_hs.build_vocab(new_sentences, update=True)
        self.assertEqual(len(model_hs.wv), 14)
        self.assertEqual(model_hs.wv.get_vecattr('graph', 'count'), 4)
        self.assertEqual(model_hs.wv.get_vecattr('artificial', 'count'), 4)

    def test_online_learning_fromfile(self):
        with temporary_file('gensim_fasttext1.tst') as corpus_file, temporary_file('gensim_fasttext2.tst') as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)
            model_hs = FT_gensim(corpus_file=corpus_file, vector_size=12, min_count=1, seed=42, hs=1, negative=0, bucket=BUCKET)
            self.assertTrue(len(model_hs.wv), 12)
            self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 3)
            model_hs.build_vocab(corpus_file=new_corpus_file, update=True)
            self.assertEqual(len(model_hs.wv), 14)
            self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 4)
            self.assertTrue(model_hs.wv.get_vecattr('artificial', 'count'), 4)

    def test_online_learning_after_save(self):
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model_neg = FT_gensim(sentences, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
        model_neg.save(tmpf)
        model_neg = FT_gensim.load(tmpf)
        self.assertTrue(len(model_neg.wv), 12)
        model_neg.build_vocab(new_sentences, update=True)
        model_neg.train(new_sentences, total_examples=model_neg.corpus_count, epochs=model_neg.epochs)
        self.assertEqual(len(model_neg.wv), 14)

    def test_online_learning_through_ft_format_saves(self):
        tmpf = get_tmpfile('gensim_ft_format.tst')
        model = FT_gensim(sentences, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
        gensim.models.fasttext.save_facebook_model(model, tmpf)
        model_reload = gensim.models.fasttext.load_facebook_model(tmpf)
        self.assertTrue(len(model_reload.wv), 12)
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors))
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors_vocab))
        model_reload.build_vocab(new_sentences, update=True)
        model_reload.train(new_sentences, total_examples=model_reload.corpus_count, epochs=model_reload.epochs)
        self.assertEqual(len(model_reload.wv), 14)
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors))
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors_vocab))
        tmpf2 = get_tmpfile('gensim_ft_format2.tst')
        gensim.models.fasttext.save_facebook_model(model_reload, tmpf2)

    def test_online_learning_after_save_fromfile(self):
        with temporary_file('gensim_fasttext1.tst') as corpus_file, temporary_file('gensim_fasttext2.tst') as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)
            tmpf = get_tmpfile('gensim_fasttext.tst')
            model_neg = FT_gensim(corpus_file=corpus_file, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
            model_neg.save(tmpf)
            model_neg = FT_gensim.load(tmpf)
            self.assertTrue(len(model_neg.wv), 12)
            model_neg.build_vocab(corpus_file=new_corpus_file, update=True)
            model_neg.train(corpus_file=new_corpus_file, total_words=model_neg.corpus_total_words, epochs=model_neg.epochs)
            self.assertEqual(len(model_neg.wv), 14)

    def online_sanity(self, model):
        terro, others = ([], [])
        for line in list_corpus:
            if 'terrorism' in line:
                terro.append(line)
            else:
                others.append(line)
        self.assertTrue(all(('terrorism' not in line for line in others)))
        model.build_vocab(others)
        start_vecs = model.wv.vectors_vocab.copy()
        model.train(others, total_examples=model.corpus_count, epochs=model.epochs)
        self.assertFalse(np.all(np.equal(start_vecs, model.wv.vectors_vocab)))
        self.assertFalse(np.all(np.equal(model.wv.vectors, model.wv.vectors_vocab)))
        self.assertFalse('terrorism' in model.wv.key_to_index)
        model.build_vocab(terro, update=True)
        self.assertTrue(model.wv.vectors_ngrams.dtype == 'float32')
        self.assertTrue('terrorism' in model.wv.key_to_index)
        orig0_all = np.copy(model.wv.vectors_ngrams)
        model.train(terro, total_examples=len(terro), epochs=model.epochs)
        self.assertFalse(np.allclose(model.wv.vectors_ngrams, orig0_all))
        sim = model.wv.n_similarity(['war'], ['terrorism'])
        assert abs(sim) > 0.6

    def test_sg_hs_online(self):
        model = FT_gensim(sg=1, window=2, hs=1, negative=0, min_count=3, epochs=1, seed=42, workers=1, bucket=BUCKET)
        self.online_sanity(model)

    def test_sg_neg_online(self):
        model = FT_gensim(sg=1, window=2, hs=0, negative=5, min_count=3, epochs=1, seed=42, workers=1, bucket=BUCKET)
        self.online_sanity(model)

    def test_cbow_hs_online(self):
        model = FT_gensim(sg=0, cbow_mean=1, alpha=0.05, window=2, hs=1, negative=0, min_count=3, epochs=1, seed=42, workers=1, bucket=BUCKET)
        self.online_sanity(model)

    def test_cbow_neg_online(self):
        model = FT_gensim(sg=0, cbow_mean=1, alpha=0.05, window=2, hs=0, negative=5, min_count=5, epochs=1, seed=42, workers=1, sample=0, bucket=BUCKET)
        self.online_sanity(model)

    def test_get_vocab_word_vecs(self):
        model = FT_gensim(vector_size=12, min_count=1, seed=42, bucket=BUCKET)
        model.build_vocab(sentences)
        original_syn0_vocab = np.copy(model.wv.vectors_vocab)
        model.wv.adjust_vectors()
        self.assertTrue(np.all(np.equal(model.wv.vectors_vocab, original_syn0_vocab)))

    def test_persistence_word2vec_format(self):
        """Test storing/loading the model in word2vec format."""
        tmpf = get_tmpfile('gensim_fasttext_w2v_format.tst')
        model = FT_gensim(sentences, min_count=1, vector_size=12, bucket=BUCKET)
        model.wv.save_word2vec_format(tmpf, binary=True)
        loaded_model_kv = KeyedVectors.load_word2vec_format(tmpf, binary=True)
        self.assertEqual(len(model.wv), len(loaded_model_kv))
        self.assertTrue(np.allclose(model.wv['human'], loaded_model_kv['human']))

    def test_bucket_ngrams(self):
        model = FT_gensim(vector_size=12, min_count=1, bucket=20)
        model.build_vocab(sentences)
        self.assertEqual(model.wv.vectors_ngrams.shape, (20, 12))
        model.build_vocab(new_sentences, update=True)
        self.assertEqual(model.wv.vectors_ngrams.shape, (20, 12))

    def test_estimate_memory(self):
        model = FT_gensim(sg=1, hs=1, vector_size=12, negative=5, min_count=3, bucket=BUCKET)
        model.build_vocab(sentences)
        report = model.estimate_memory()
        self.assertEqual(report['vocab'], 2800)
        self.assertEqual(report['syn0_vocab'], 192)
        self.assertEqual(report['syn1'], 192)
        self.assertEqual(report['syn1neg'], 192)
        self.assertEqual(report['syn0_ngrams'], model.vector_size * np.dtype(np.float32).itemsize * BUCKET)
        self.assertEqual(report['buckets_word'], 688)
        self.assertEqual(report['total'], 484064)

    def obsolete_testLoadOldModel(self):
        """Test loading fasttext models from previous version"""
        model_file = 'fasttext_old'
        model = FT_gensim.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv) == 12)
        self.assertTrue(len(model.wv.index_to_key) == 12)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.wv.vectors_lockf.shape == (12,))
        self.assertTrue(model.cum_table.shape == (12,))
        self.assertEqual(model.wv.vectors_vocab.shape, (12, 100))
        self.assertEqual(model.wv.vectors_ngrams.shape, (2000000, 100))
        model_file = 'fasttext_old_sep'
        model = FT_gensim.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv) == 12)
        self.assertTrue(len(model.wv.index_to_key) == 12)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.wv.vectors_lockf.shape == (12,))
        self.assertTrue(model.cum_table.shape == (12,))
        self.assertEqual(model.wv.vectors_vocab.shape, (12, 100))
        self.assertEqual(model.wv.vectors_ngrams.shape, (2000000, 100))

    def test_vectors_for_all_with_inference(self):
        """Test vectors_for_all can infer new vectors."""
        words = ['responding', 'approached', 'chairman', 'an out-of-vocabulary word', 'another out-of-vocabulary word']
        vectors_for_all = self.test_model.wv.vectors_for_all(words)
        expected = 5
        predicted = len(vectors_for_all)
        assert expected == predicted
        expected = self.test_model.wv['responding']
        predicted = vectors_for_all['responding']
        assert np.allclose(expected, predicted)
        smaller_distance = np.linalg.norm(vectors_for_all['an out-of-vocabulary word'] - vectors_for_all['another out-of-vocabulary word'])
        greater_distance = np.linalg.norm(vectors_for_all['an out-of-vocabulary word'] - vectors_for_all['responding'])
        assert greater_distance > smaller_distance

    def test_vectors_for_all_without_inference(self):
        """Test vectors_for_all does not infer new vectors when prohibited."""
        words = ['responding', 'approached', 'chairman', 'an out-of-vocabulary word', 'another out-of-vocabulary word']
        vectors_for_all = self.test_model.wv.vectors_for_all(words, allow_inference=False)
        expected = 3
        predicted = len(vectors_for_all)
        assert expected == predicted
        expected = self.test_model.wv['responding']
        predicted = vectors_for_all['responding']
        assert np.allclose(expected, predicted)

    def test_negative_ns_exp(self):
        """The model should accept a negative ns_exponent as a valid value."""
        model = FT_gensim(sentences, ns_exponent=-1, min_count=1, workers=1)
        tmpf = get_tmpfile('fasttext_negative_exp.tst')
        model.save(tmpf)
        loaded_model = FT_gensim.load(tmpf)
        loaded_model.train(sentences, total_examples=model.corpus_count, epochs=1)
        assert loaded_model.ns_exponent == -1, loaded_model.ns_exponent