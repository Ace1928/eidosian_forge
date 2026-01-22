import math
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
def _sample_with_step(self, prob):
    """Sample a value with the cumulative prob in the given range.

        The range is divided evenly by `step`. So only sampling from a finite
        set of values. When calling the function, no need to use (max_value + 1)
        since the function takes care of the inclusion of max_value.
        """
    n_values = self._get_n_values()
    index = hp_utils.prob_to_index(prob, n_values)
    return self._get_value_by_index(index)