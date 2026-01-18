from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_treebank_span_tokenizer(self):
    """
        Test TreebankWordTokenizer.span_tokenize function
        """
    tokenizer = TreebankWordTokenizer()
    test1 = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).'
    expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23), (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38), (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59), (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]
    result = list(tokenizer.span_tokenize(test1))
    assert result == expected
    test2 = 'The DUP is similar to the "religious right" in the United States and takes a hardline stance on social issues'
    expected = [(0, 3), (4, 7), (8, 10), (11, 18), (19, 21), (22, 25), (26, 27), (27, 36), (37, 42), (42, 43), (44, 46), (47, 50), (51, 57), (58, 64), (65, 68), (69, 74), (75, 76), (77, 85), (86, 92), (93, 95), (96, 102), (103, 109)]
    result = list(tokenizer.span_tokenize(test2))
    assert result == expected
    test3 = 'The DUP is similar to the "religious right" in the United States and takes a ``hardline\'\' stance on social issues'
    expected = [(0, 3), (4, 7), (8, 10), (11, 18), (19, 21), (22, 25), (26, 27), (27, 36), (37, 42), (42, 43), (44, 46), (47, 50), (51, 57), (58, 64), (65, 68), (69, 74), (75, 76), (77, 79), (79, 87), (87, 89), (90, 96), (97, 99), (100, 106), (107, 113)]
    result = list(tokenizer.span_tokenize(test3))
    assert result == expected