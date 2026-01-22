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
class VariableScopeLayer(base_layer.Layer):
    """Wrapper Layer to capture `compat.v1.get_variable` and `compat.v1.layers`.

    This shim layer allows using large sets of TF1 model-forward-pass code as a
    Keras layer that works in TF2 with TF2 behaviors enabled. It will capture
    both weights and regularization losses of your forward-pass code. To use it,
    override this class and put your TF1 model's forward pass inside your
    implementation for `forward_pass`. (Unlike standard custom Keras layers,
    do not override `call`.)

    Below are some examples, and then more details on the functionality of this
    shim layer to wrap TF1 model forward passes.

    Example of capturing tf.compat.v1.layer-based modeling code as a Keras
    layer:

    ```python
    class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs):
        with variable_scope.variable_scope("double_dense_layer"):
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
    class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs):
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
      `compat.v1.layer` creations will get captured by this wrapper layer.
      Regularization losses are accessible in `layer.losses` after a call just
      like in a standard Keras layer, and will be captured by any model that
      includes this layer.  Regularization losses attached to Keras
      layers/models set as attributes of your layer will also get captured in
      the standard Keras regularization loss tracking.

    Variable scope / variable reuse:
      variable-scope based reuse in the `forward_pass` will be respected,
      and work like variable-scope based reuse in TF1.

    Variable Names/Pre-trained checkpoint loading:
      Variable naming from get_variable and `compat.v1.layer` layers will match
      the TF1 names, so you should be able to re-use your old name-based
      checkpoints. Variable naming for Keras layers/models or for variables
      created by `tf.Variable` may change when going to eager execution.

    Training Arg in `forward_pass`:
      Keras will pass a `training` arg to this layer if `forward_pass` contains
      a `training` arg or a `**kwargs` varargs in its call signature,
      similarly to how keras passes `training` to other layers in TF2 that have
      similar signatures in their `call` implementations.
      See more details in the docs
      on `tf.keras.layers.Layer` to understand what will be passed and when.
      Note: tf.compat.v1.layers are usually not called with `training=None`,
      so the training arg to `forward_pass` might not feed through to them
      unless you pass it to their calls explicitly.

    Call signature of the forward pass:
      The semantics of the forward pass signature match the standard
      Keras layer `call` signature, including how Keras decides when
      to pass in a `training` arg., and the semantics applied to
      the first positional arg in the call signature.

    Caveats:
      * TF2 will not prune unused variable updates (or unused outputs). You may
        need to adjust your forward pass code to avoid computations or variable
        updates that you don't intend to use. (E.g. by adding a flag to the
        `forward_pass` call signature and branching on it).
      * Avoid Nesting variable creation in tf.function inside of `forward_pass`
        While the layer may safely be used from inside a `tf.function`, using
        a function inside of `forward_pass` will break the variable scoping.
      * If you would like to nest Keras layers/models or other
        `VariableScopeLayer`s directly in `forward_pass`, you need to
        assign them as attributes of your layer so that Keras's standard
        object-oriented weights and loss tracking will kick in.
        See the intro to modules, layers, and models
        [guide](https://www.tensorflow.org/guide/intro_to_modules) for more info
    """

    @property
    @layer_utils.cached_per_instance
    def _call_full_argspec(self):
        return tf_inspect.getfullargspec(self.forward_pass)

    def forward_pass(self, *args, **kwargs):
        """Implement this method. It should include your model forward pass."""
        raise NotImplementedError

    @track_tf1_style_variables
    def call(self, *args, **kwargs):
        return self.forward_pass(*args, **kwargs)