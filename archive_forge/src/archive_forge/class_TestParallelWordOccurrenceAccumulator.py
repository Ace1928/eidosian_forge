import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
class TestParallelWordOccurrenceAccumulator(BaseTestCases.TextAnalyzerTestBase):
    accumulator_cls = ParallelWordOccurrenceAccumulator

    def init_accumulator(self):
        return self.accumulator_cls(2, self.top_ids, self.dictionary)

    def init_accumulator2(self):
        return self.accumulator_cls(2, self.top_ids2, self.dictionary2)