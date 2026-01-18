from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_word_tokenize(self):
    """
        Test word_tokenize function
        """
    sentence = "The 'v', I've been fooled but I'll seek revenge."
    expected = ['The', "'", 'v', "'", ',', 'I', "'ve", 'been', 'fooled', 'but', 'I', "'ll", 'seek', 'revenge', '.']
    assert word_tokenize(sentence) == expected
    sentence = "'v' 're'"
    expected = ["'", 'v', "'", "'re", "'"]
    assert word_tokenize(sentence) == expected