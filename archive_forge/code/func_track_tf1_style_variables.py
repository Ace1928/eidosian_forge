from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import functools
import tensorflow.compat.v2 as tf
from keras.src.engine import base_layer
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
@keras_export(v1=['keras.utils.track_tf1_style_variables'])
def track_tf1_style_variables(method):
    """Wrap layer & module methods in this decorator to capture tf1-style
    weights.

    Decorating a `tf.keras.Layer`'s  or `tf.Module`'s methods with this
    decorator will cause the layer/module to track weights created/used
    via `tf.compat.v1.get_variable` (and by extension `tf.compat.v1.layers`)
    inside the decorated method.

    In addition to tracking the weights themselves under the standard
    `layer.variable`/`module.variable`/etc. properties, if the method belongs
    to a `tf.keras.Layer` then any regularization losses specified via the
    `get_variable` or `tf.compat.v1.layers` regularizer arguments will get
    tracked by the layer under the standard `layer.losses` property.

    This tracking enables using large classes of TF1-style model-forward-pass
    code inside of Keras layers or `tf.Modules` in TF2 with TF2 behaviors
    enabled.

    Example of capturing tf.compat.v1.layer-based modeling code as a Keras
    layer:

    ```python
    class WrappedDoubleDenseLayer(tf.keras.layers.Layer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      @tf.compat.v1.keras.utils.track_tf1_style_variables
      def call(self, inputs):
        with tf.compat.v1.variable_scope("double_dense_layer"):
          out = tf.compat.v1.layers.dense(
              inputs, self.units, name="dense_one",
              kernel_initializer=tf.compat.v1.random_normal_initializer,
              kernel_regularizer="l2")
          out = tf.compat.v1.layers.dense(
              out, self.units, name="dense_two",
              kernel_initializer=tf.compat.v1.random_normal_initializer(),
              kernel_regularizer="l2")
        return out

    # Create a layer that can be used as a standard keras layer
    layer = WrappedDoubleDenseLayer(10)

    # call the layer on inputs
    layer(...)

    # Variables created/used within the scope will be tracked by the layer
    layer.weights
    layer.trainable_variables

    # Regularization losses will be captured in layer.losses after a call,
    # just like any other Keras layer
    reg_losses = layer.losses
    ```

    Example of capturing tf.compat.v1.get_variable-based modeling code as
    a Keras layer:

    ```python
    class WrappedDoubleDenseLayer(tf.keras.layers.Layer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      @tf.compat.v1.keras.utils.track_tf1_style_variables
      def call(self, inputs):
        out = inputs
        with tf.compat.v1.variable_scope("double_dense_layer"):
          with tf.compat.v1.variable_scope("dense_one"):
            # The weights are created with a `regularizer`,
            # so the layer should track their regularization losses
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=init_ops.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=init_ops.zeros_initializer(),
                name="bias")
            out = tf.compat.v1.math.matmul(out, kernel)
            out = tf.compat.v1.nn.bias_add(out, bias)
          with tf.compat.v1.variable_scope("dense_two"):
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=init_ops.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=init_ops.zeros_initializer(),
                name="bias")
            out = tf.compat.v1.math.matmul(out, kernel)
            out = tf.compat.v1.nn.bias_add(out, bias)
        return out

    # Create a layer that can be used as a standard keras layer
    layer = WrappedDoubleDenseLayer(10)

    # call the layer on inputs
    layer(...)

    # Variables created/used within the scope will be tracked by the layer
    layer.weights
    layer.trainable_variables

    # Regularization losses will be captured in layer.losses after a call,
    # just like any other Keras layer
    reg_losses = layer.losses
    ```

    Regularization losses:
      Any regularizers specified in the `get_variable` calls or
      `compat.v1.layer` creations will get captured if they occur in your
      decorated method and the method belongs to a
      `tf.keras.Layer`/`tf.keras.Module`. Regularization losses
      are accessible in `layer.losses` after a call just like in a standard
      Keras layer, and will be captured by any model that includes this layer.
      Regularization losses attached to Keras layers/models set as attributes
      of your layer will also get captured in the standard Keras regularization
      loss tracking.

      (While Modules have no `losses` property, no-arg callables to compute
       the regularization losses may be tracked as dict values in a private
       `module._tf1_style_var_store._regularizers` property, but only for
       `tf.compat.v1.layers` and `get_variable` weights and not for any other
       nested Keras layers/tf.Modules)

    Variable scope / variable reuse:
      variable-scope based reuse in your decorated method will be respected,
      and work like variable-scope based reuse in TF1.

    Variable Names/Pre-trained checkpoint loading:
      Variable naming from get_variable and `compat.v1.layer` layers will match
      the TF1 names, so you should be able to re-use your old name-based
      checkpoints. Variable naming for Keras layers/models or for variables
      created by `tf.Variable` may change when going to eager execution.

    Training Arg if you decorate `layer.call`:
      Keras will pass a `training` arg to this layer if `call` contains
      a `training` arg or a `**kwargs` varargs in its call signature,
      similarly to how keras passes `training` to other layers in TF2 that have
      similar signatures in their `call` implementations.
      See more details in the docs
      on `tf.keras.layers.Layer` to understand what will be passed and when.
      Note: tf.compat.v1.layers are usually not called with `training=None`,
      so the training arg to `forward_pass` might not feed through to them
      unless you pass it to their calls explicitly.

    Caveats:
      * TF2 will not prune unused variable updates (or unused outputs). You may
        need to adjust your forward pass code to avoid computations or variable
        updates that you don't intend to use.
      * Avoid Nesting variable creation in tf.function inside of
        methods decorated with `track_tf1_style_variables`
        While the method may safely be used from inside a `tf.function`, using
        a function inside of a decorated method may break the variable scoping.
      * This decorator only adds implicit tracking for legacy tf1-style
        get_variable / compat.v1.layers usage.
        If you would like to use nested Keras layers/models
        inside the decorated method, you need to
        assign them as attributes of your layer so that Keras/Module's standard
        object-oriented weights (and loss tracking for layers) will kick in.
        See the intro to modules, layers, and models
        [guide](https://www.tensorflow.org/guide/intro_to_modules) for more
        info.  As a backup, the `compat.v1.keras.utils.get_or_create_layer`
        method will ease tracking nested keras model weights and losses for
        existing TF1 code, but new code should use explicit tracking.

    Args:
      method: The method to decorate. This should belong to a custom tf.Module,
      tf.keras.layers.Layer, or tf.keras.Model.

    Returns:
      The decorated method.
    """

    def _method_wrapper(self, *args, **kwargs):
        var_store = getattr(self, '_tf1_style_var_store', None)
        if not var_store:
            if not isinstance(self, tf.Module):
                raise ValueError('`@tf.compat.v1.keras.utils.track_tf1_layers_and_variables` must be applied to a method of a subclassed `tf.Module`, `tf.keras.layers.Layer`, or `tf.keras.Model` and which takes `self` as the first argument. But, the first argument passed to the decorated method was {}, which does not extend Module, Layer, or Model.'.format(self))
            var_store = _EagerVariableStore()
            self._tf1_style_var_store = var_store
        existing_regularized_variables = set(var_store._regularizers.keys())
        with var_store.scope():
            out = method(self, *args, **kwargs)
        if isinstance(self, base_layer.Layer):
            for var_name, regularizer in var_store._regularizers.items():
                if var_name not in existing_regularized_variables:
                    self.add_loss(regularizer)
        return out
    return tf.__internal__.decorator.make_decorator(target=method, decorator_func=_method_wrapper)