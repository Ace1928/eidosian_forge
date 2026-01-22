import math
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
def _numerical_to_prob(self, value, max_value=None):
    """Convert a numerical value to range [0.0, 1.0)."""
    if max_value is None:
        max_value = self.max_value
    if max_value == self.min_value:
        return 0.5
    if self.sampling == 'linear':
        return (value - self.min_value) / (max_value - self.min_value)
    if self.sampling == 'log':
        return math.log(value / self.min_value) / math.log(max_value / self.min_value)
    if self.sampling == 'reverse_log':
        return 1.0 - math.log((max_value + self.min_value - value) / self.min_value) / math.log(max_value / self.min_value)