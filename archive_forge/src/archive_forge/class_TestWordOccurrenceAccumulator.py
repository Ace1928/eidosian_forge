import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
class TestWordOccurrenceAccumulator(BaseTestCases.TextAnalyzerTestBase):
    accumulator_cls = WordOccurrenceAccumulator