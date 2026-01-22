import math
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
def _sample_numerical_value(self, prob, max_value=None):
    """Sample a value with the cumulative prob in the given range."""
    if max_value is None:
        max_value = self.max_value
    if self.sampling == 'linear':
        return prob * (max_value - self.min_value) + self.min_value
    elif self.sampling == 'log':
        return self.min_value * math.pow(max_value / self.min_value, prob)
    elif self.sampling == 'reverse_log':
        return max_value + self.min_value - self.min_value * math.pow(max_value / self.min_value, 1 - prob)