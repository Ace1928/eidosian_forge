import math
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
def _get_n_values(self):
    """Get the total number of possible values using step."""
    if self.sampling == 'linear':
        return int((self.max_value - self.min_value) // self.step + 1)
    return int(math.log(self.max_value / self.min_value, self.step) + 1e-08) + 1