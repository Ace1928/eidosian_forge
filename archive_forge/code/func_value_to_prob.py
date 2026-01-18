from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters.hp_types import numerical
def value_to_prob(self, value):
    if self.step is None:
        return self._numerical_to_prob(value + 0.5, self.max_value + 1)
    return self._to_prob_with_step(value)