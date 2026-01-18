from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_syllable_tokenizer_numbers(self):
    """
        Test SyllableTokenizer tokenizer.
        """
    tokenizer = SyllableTokenizer()
    text = '9' * 10000
    tokens = tokenizer.tokenize(text)
    assert tokens == [text]