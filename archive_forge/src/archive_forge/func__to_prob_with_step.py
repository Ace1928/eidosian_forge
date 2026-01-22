import math
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
def _to_prob_with_step(self, value):
    """Convert to cumulative prob with step specified.

        When calling the function, no need to use (max_value + 1) since the
        function takes care of the inclusion of max_value.
        """
    if self.sampling == 'linear':
        index = (value - self.min_value) // self.step
    if self.sampling == 'log':
        index = math.log(value / self.min_value, self.step)
    if self.sampling == 'reverse_log':
        index = math.log((self.max_value - value + self.min_value) / self.min_value, self.step)
    n_values = self._get_n_values()
    return hp_utils.index_to_prob(index, n_values)