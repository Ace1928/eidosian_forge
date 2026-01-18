import numpy
import pytest
from thinc import registry
from thinc.api import (
@pytest.mark.parametrize('guesses, labels, names', [([guesses1, guesses2], [labels1, labels2], []), ([guesses1, guesses2], [labels1_full, labels2], []), ([guesses1, guesses2], [labels1_strings, labels2_strings], ['A', 'B', 'C'])])
def test_sequence_categorical_crossentropy(guesses, labels, names):
    d_scores = SequenceCategoricalCrossentropy(normalize=False, names=names).get_grad(guesses, labels)
    d_scores1 = d_scores[0]
    d_scores2 = d_scores[1]
    assert d_scores1.shape == guesses1.shape
    assert d_scores2.shape == guesses2.shape
    assert d_scores1[1][0] == pytest.approx(0.4, eps)
    assert d_scores1[1][1] == pytest.approx(-0.4, eps)
    d_scores = SequenceCategoricalCrossentropy(normalize=True, names=names).get_grad(guesses, labels)
    d_scores1 = d_scores[0]
    d_scores2 = d_scores[1]
    assert d_scores1[1][0] == pytest.approx(0.2, eps)
    assert d_scores1[1][1] == pytest.approx(-0.2, eps)
    assert d_scores1[2][0] == pytest.approx(0, eps)
    assert d_scores1[2][1] == pytest.approx(0.5, eps)
    assert d_scores1[2][2] == pytest.approx(0.5, eps)
    assert d_scores1[3][0] == pytest.approx(0, eps)
    assert d_scores1[3][1] == pytest.approx(0, eps)
    assert d_scores1[3][2] == pytest.approx(-0.5, eps)
    assert d_scores2[0][0] == pytest.approx(0.1, eps)
    assert d_scores2[0][1] == pytest.approx(-0.35, eps)
    loss = SequenceCategoricalCrossentropy(normalize=True, names=names).get_loss(guesses, labels)
    assert loss == pytest.approx(1.09, eps)