from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters.hp_types import numerical
def _check_int(val, arg):
    int_val = int(val)
    if int_val != val:
        raise ValueError(f'{arg} must be an int, Received: {str(val)} of type {type(val)}.')
    return int_val