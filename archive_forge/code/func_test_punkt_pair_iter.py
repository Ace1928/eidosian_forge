from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_punkt_pair_iter(self):
    test_cases = [('12', [('1', '2'), ('2', None)]), ('123', [('1', '2'), ('2', '3'), ('3', None)]), ('1234', [('1', '2'), ('2', '3'), ('3', '4'), ('4', None)])]
    for test_input, expected_output in test_cases:
        actual_output = [x for x in punkt._pair_iter(test_input)]
        assert actual_output == expected_output