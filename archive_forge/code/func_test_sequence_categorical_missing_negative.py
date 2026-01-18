import numpy
import pytest
from thinc import registry
from thinc.api import (
@pytest.mark.parametrize('guesses, labels, names', [([guesses1], [['A', '!A', '', '!C']], ['A', 'B', 'C'])])
def test_sequence_categorical_missing_negative(guesses, labels, names):
    d_scores = SequenceCategoricalCrossentropy(normalize=False, names=names, neg_prefix='!', missing_value='').get_grad(guesses, labels)
    d_scores0 = d_scores[0]
    assert d_scores0[0][0] == pytest.approx(-0.9, eps)
    assert d_scores0[0][1] == pytest.approx(0.5, eps)
    assert d_scores0[0][2] == pytest.approx(0.6, eps)
    assert d_scores0[1][0] == pytest.approx(0.4, eps)
    assert d_scores0[1][1] == pytest.approx(0.0, eps)
    assert d_scores0[1][2] == pytest.approx(0.0, eps)
    assert d_scores0[2][0] == pytest.approx(0.0, eps)
    assert d_scores0[2][1] == pytest.approx(0.0, eps)
    assert d_scores0[2][2] == pytest.approx(0.0, eps)
    assert d_scores0[3][0] == pytest.approx(0.0, eps)
    assert d_scores0[3][1] == pytest.approx(0.0, eps)
    assert d_scores0[3][2] == pytest.approx(0.0, eps)