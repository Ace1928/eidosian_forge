import numpy as np
import pytest
from sklearn.utils._plotting import _interval_max_min_ratio, _validate_score_name
@pytest.mark.parametrize('score_name, scoring, negate_score, expected_score_name', [('accuracy', None, False, 'accuracy'), (None, 'accuracy', False, 'Accuracy'), (None, 'accuracy', True, 'Negative accuracy'), (None, 'neg_mean_absolute_error', False, 'Negative mean absolute error'), (None, 'neg_mean_absolute_error', True, 'Mean absolute error'), ('MAE', 'neg_mean_absolute_error', True, 'MAE'), (None, None, False, 'Score'), (None, None, True, 'Negative score'), ('Some metric', metric, False, 'Some metric'), ('Some metric', metric, True, 'Some metric'), (None, metric, False, 'Metric'), (None, metric, True, 'Negative metric'), ('Some metric', neg_metric, False, 'Some metric'), ('Some metric', neg_metric, True, 'Some metric'), (None, neg_metric, False, 'Negative metric'), (None, neg_metric, True, 'Metric')])
def test_validate_score_name(score_name, scoring, negate_score, expected_score_name):
    """Check that we return the right score name."""
    assert _validate_score_name(score_name, scoring, negate_score) == expected_score_name