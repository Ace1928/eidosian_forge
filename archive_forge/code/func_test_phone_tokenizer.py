from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_phone_tokenizer(self):
    """
        Test a string that resembles a phone number but contains a newline
        """
    tokenizer = TweetTokenizer()
    test1 = '(393)  928 -3010'
    expected = ['(393)  928 -3010']
    result = tokenizer.tokenize(test1)
    assert result == expected
    test2 = '(393)\n928 -3010'
    expected = ['(', '393', ')', '928 -3010']
    result = tokenizer.tokenize(test2)
    assert result == expected