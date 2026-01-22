from __future__ import division  # always use floats
from __future__ import with_statement
import logging
import os
import unittest
from gensim import utils, corpora, models, similarities
from gensim.test.utils import datapath, get_tmpfile
class TestMiislita(unittest.TestCase):

    def test_textcorpus(self):
        """Make sure TextCorpus can be serialized to disk. """
        miislita = CorpusMiislita(datapath('head500.noblanks.cor.bz2'))
        ftmp = get_tmpfile('test_textcorpus.mm')
        corpora.MmCorpus.save_corpus(ftmp, miislita)
        self.assertTrue(os.path.exists(ftmp))
        miislita2 = corpora.MmCorpus(ftmp)
        self.assertEqual(list(miislita), list(miislita2))

    def test_save_load_ability(self):
        """
        Make sure we can save and load (un/pickle) TextCorpus objects (as long
        as the underlying input isn't a file-like object; we cannot pickle those).
        """
        corpusname = datapath('miIslita.cor')
        miislita = CorpusMiislita(corpusname)
        tmpf = get_tmpfile('tc_test.cpickle')
        miislita.save(tmpf)
        miislita2 = CorpusMiislita.load(tmpf)
        self.assertEqual(len(miislita), len(miislita2))
        self.assertEqual(miislita.dictionary.token2id, miislita2.dictionary.token2id)

    def test_miislita_high_level(self):
        miislita = CorpusMiislita(datapath('miIslita.cor'))
        tfidf = models.TfidfModel(miislita, miislita.dictionary, normalize=False)
        index = similarities.SparseMatrixSimilarity(tfidf[miislita], num_features=len(miislita.dictionary))
        query = 'latent semantic indexing'
        vec_bow = miislita.dictionary.doc2bow(query.lower().split())
        vec_tfidf = tfidf[vec_bow]
        sims_tfidf = index[vec_tfidf]
        expected = [0.0, 0.256, 0.7022, 0.1524, 0.3334]
        for i, value in enumerate(expected):
            self.assertAlmostEqual(sims_tfidf[i], value, 2)