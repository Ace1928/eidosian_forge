import functools
from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils
class KerasModelTypeCombination(test_combinations.TestCombination):
    """Combination for Keras model types when doing model test.

  It by default includes 'functional', 'subclass', 'sequential'.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.
  """

    def context_managers(self, kwargs):
        model_type = kwargs.pop('model_type', None)
        if model_type in KERAS_MODEL_TYPES:
            return [testing_utils.model_type_scope(model_type)]
        else:
            return []

    def parameter_modifiers(self):
        return [test_combinations.OptionalParameter('model_type')]