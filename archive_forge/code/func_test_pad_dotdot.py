from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_pad_dotdot(self):
    """
        Test padding of dotdot* for word tokenization.
        """
    text = 'Why did dotdot.. not get tokenized but dotdotdot... did? How about manydots.....'
    expected = ['Why', 'did', 'dotdot', '..', 'not', 'get', 'tokenized', 'but', 'dotdotdot', '...', 'did', '?', 'How', 'about', 'manydots', '.....']
    assert word_tokenize(text) == expected