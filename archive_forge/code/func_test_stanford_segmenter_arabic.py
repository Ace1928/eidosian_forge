from typing import List, Tuple
import pytest
from nltk.tokenize import (
@check_stanford_segmenter
def test_stanford_segmenter_arabic(self):
    """
        Test the Stanford Word Segmenter for Arabic (default config)
        """
    seg = StanfordSegmenter()
    seg.default_config('ar')
    sent = 'يبحث علم الحاسوب استخدام الحوسبة بجميع اشكالها لحل المشكلات'
    segmented_sent = seg.segment(sent.split())
    assert segmented_sent.split() == ['يبحث', 'علم', 'الحاسوب', 'استخدام', 'الحوسبة', 'ب', 'جميع', 'اشكال', 'ها', 'ل', 'حل', 'المشكلات']