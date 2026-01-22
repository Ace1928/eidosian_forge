import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class PhrasesCommon(PhrasesData):
    """Tests for both Phrases and FrozenPhrases classes."""

    def setUp(self):
        self.bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        self.bigram_default = Phrases(self.sentences, connector_words=self.connector_words)

    def test_empty_phrasified_sentences_iterator(self):
        bigram_phrases = Phrases(self.sentences)
        bigram_phraser = FrozenPhrases(bigram_phrases)
        trigram_phrases = Phrases(bigram_phraser[self.sentences])
        trigram_phraser = FrozenPhrases(trigram_phrases)
        trigrams = trigram_phraser[bigram_phraser[self.sentences]]
        fst, snd = (list(trigrams), list(trigrams))
        self.assertEqual(fst, snd)
        self.assertNotEqual(snd, [])

    def test_empty_inputs_on_bigram_construction(self):
        """Test that empty inputs don't throw errors and return the expected result."""
        self.assertEqual(list(self.bigram_default[[]]), [])
        self.assertEqual(list(self.bigram_default[iter(())]), [])
        self.assertEqual(list(self.bigram_default[[[], []]]), [[], []])
        self.assertEqual(list(self.bigram_default[iter([[], []])]), [[], []])
        self.assertEqual(list(self.bigram_default[(iter(()) for i in range(2))]), [[], []])

    def test_sentence_generation(self):
        """Test basic bigram using a dummy corpus."""
        self.assertEqual(len(self.sentences), len(list(self.bigram_default[self.sentences])))

    def test_sentence_generation_with_generator(self):
        """Test basic bigram production when corpus is a generator."""
        self.assertEqual(len(list(self.gen_sentences())), len(list(self.bigram_default[self.gen_sentences()])))

    def test_bigram_construction(self):
        """Test Phrases bigram construction."""
        bigram1_seen = False
        bigram2_seen = False
        for sentence in self.bigram[self.sentences]:
            if not bigram1_seen and self.bigram1 in sentence:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in sentence:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[1]])
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[4]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-2]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-1]])
        self.assertTrue(self.bigram3 in self.bigram[self.sentences[-1]])

    def test_bigram_construction_from_generator(self):
        """Test Phrases bigram construction building when corpus is a generator."""
        bigram1_seen = False
        bigram2_seen = False
        for s in self.bigram[self.gen_sentences()]:
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)

    def test_bigram_construction_from_array(self):
        """Test Phrases bigram construction building when corpus is a numpy array."""
        bigram1_seen = False
        bigram2_seen = False
        for s in self.bigram[np.array(self.sentences, dtype=object)]:
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)