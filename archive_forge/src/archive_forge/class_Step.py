from tensorflow.python.eager import backprop
from tensorflow.python.training import optimizer as optimizer_lib
class Step(object):
    """Interface for performing each step of a training algorithm."""

    def __init__(self, distribution):
        self._distribution = distribution

    @property
    def distribution(self):
        return self._distribution

    def initialize(self):
        return []

    def __call__(self):
        """Perform one step of this training algorithm."""
        raise NotImplementedError('must be implemented in descendants')