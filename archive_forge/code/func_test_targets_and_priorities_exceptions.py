import pytest
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import scalarize
def test_targets_and_priorities_exceptions(self) -> None:
    targets = [1, 1]
    priorities = [1]
    with pytest.raises(AssertionError, match='Number of objectives and priorities'):
        scalarize.targets_and_priorities(self.objectives, priorities, targets)
    priorities = [1, 1]
    targets = [1]
    with pytest.raises(AssertionError, match='Number of objectives and targets'):
        scalarize.targets_and_priorities(self.objectives, priorities, targets)
    priorities = [1, 1]
    targets = [1, 1]
    limits = [1]
    with pytest.raises(AssertionError, match='Number of objectives and limits'):
        scalarize.targets_and_priorities(self.objectives, priorities, targets, limits)
    limits = [1, 1]
    off_target = -1
    with pytest.raises(AssertionError, match='The off_target argument must be nonnegative'):
        scalarize.targets_and_priorities(self.objectives, priorities, targets, limits, off_target)