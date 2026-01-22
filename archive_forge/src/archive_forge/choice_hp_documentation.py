import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
Choice of one value among a predefined set of possible values.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        values: A list of possible values. Values must be int, float,
            str, or bool. All values must be of the same type.
        ordered: Optional boolean, whether the values passed should be
            considered to have an ordering. Defaults to `True` for float/int
            values.  Must be `False` for any other values.
        default: Optional default value to return for the parameter.
            If unspecified, the default value will be:
            - None if None is one of the choices in `values`
            - The first entry in `values` otherwise.
    