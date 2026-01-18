from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_sonority_sequencing_syllable_tokenizer(self):
    """
        Test SyllableTokenizer tokenizer.
        """
    tokenizer = SyllableTokenizer()
    tokens = tokenizer.tokenize('justification')
    assert tokens == ['jus', 'ti', 'fi', 'ca', 'tion']