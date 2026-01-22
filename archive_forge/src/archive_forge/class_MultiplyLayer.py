from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
class MultiplyLayer(AssertTypeLayer):
    """A layer which multiplies its input by a scalar variable."""

    def __init__(self, regularizer=None, activity_regularizer=None, use_operator=False, var_name='v', **kwargs):
        """Initializes the MultiplyLayer.

    Args:
      regularizer: The weight regularizer on the scalar variable.
      activity_regularizer: The activity regularizer.
      use_operator: If True, add using the * operator. If False, add using
        tf.multiply.
      var_name: The name of the variable. It can be useful to pass a name other
        than 'v', to test having the attribute name (self.v) being different
        from the variable name.
      **kwargs: Passed to AssertTypeLayer constructor.
    """
        self._regularizer = regularizer
        if isinstance(regularizer, dict):
            self._regularizer = regularizers.deserialize(regularizer, custom_objects=globals())
        self._activity_regularizer = activity_regularizer
        if isinstance(activity_regularizer, dict):
            self._activity_regularizer = regularizers.deserialize(activity_regularizer, custom_objects=globals())
        self._use_operator = use_operator
        self._var_name = var_name
        super(MultiplyLayer, self).__init__(activity_regularizer=self._activity_regularizer, **kwargs)

    def build(self, _):
        self.v = self.add_weight(self._var_name, (), initializer='ones', regularizer=self._regularizer)
        self.built = True

    def call(self, inputs):
        self.assert_input_types(inputs)
        return self._multiply(inputs, self.v)

    def _multiply(self, x, y):
        if self._use_operator:
            return x * y
        else:
            return math_ops.multiply(x, y)

    def get_config(self):
        config = super(MultiplyLayer, self).get_config()
        config['regularizer'] = regularizers.serialize(self._regularizer)
        config['activity_regularizer'] = regularizers.serialize(self._activity_regularizer)
        config['use_operator'] = self._use_operator
        config['var_name'] = self._var_name
        config['assert_type'] = self._assert_type
        return config