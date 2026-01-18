import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('c', None, 1.0 / 18), ('z', None, 0.0 / 18), ('y', None, 3 / 18), ('c', ['b'], 1 / 2), ('c', ['a', 'b'], 1 / 1), ('c', ['z', 'b'], 0.4 * (1 / 2))])
def test_stupid_backoff_trigram_score(stupid_backoff_trigram_model, word, context, expected_score):
    assert pytest.approx(stupid_backoff_trigram_model.score(word, context), 0.0001) == expected_score