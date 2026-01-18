import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('c', None, 1.0 / 18), ('z', None, 0 / 18), ('y', None, 3.0 / 18), ('c', ['b'], (1 - 0.5) * 0.5 + 0.5 * 1 / 18), ('c', ['a', 'b'], 1 - 0.5 + 0.5 * ((1 - 0.5) * 0.5 + 0.5 * 1 / 18)), ('c', ['z', 'b'], (1 - 0.5) * 0.5 + 0.5 * 1 / 18)])
def test_wittenbell_trigram_score(wittenbell_trigram_model, word, context, expected_score):
    assert pytest.approx(wittenbell_trigram_model.score(word, context), 0.0001) == expected_score