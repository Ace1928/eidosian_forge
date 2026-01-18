from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_pad_asterisk(self):
    """
        Test padding of asterisk for word tokenization.
        """
    text = 'This is a, *weird sentence with *asterisks in it.'
    expected = ['This', 'is', 'a', ',', '*', 'weird', 'sentence', 'with', '*', 'asterisks', 'in', 'it', '.']
    assert word_tokenize(text) == expected