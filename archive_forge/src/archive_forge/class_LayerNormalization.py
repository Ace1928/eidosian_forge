import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.layers.LayerNormalization')
class LayerNormalization(Layer):
    """Layer normalization layer (Ba et al., 2016).

    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

    Given a tensor `inputs`, moments are calculated and normalization
    is performed across the axes specified in `axis`.

    Example:

    >>> data = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
    >>> print(data)
    tf.Tensor(
    [[ 0. 10.]
     [20. 30.]
     [40. 50.]
     [60. 70.]
     [80. 90.]], shape=(5, 2), dtype=float32)

    >>> layer = tf.keras.layers.LayerNormalization(axis=1)
    >>> output = layer(data)
    >>> print(output)
    tf.Tensor(
    [[-1. 1.]
     [-1. 1.]
     [-1. 1.]
     [-1. 1.]
     [-1. 1.]], shape=(5, 2), dtype=float32)

    Notice that with Layer Normalization the normalization happens across the
    axes *within* each example, rather than across different examples in the
    batch.

    If `scale` or `center` are enabled, the layer will scale the normalized
    outputs by broadcasting them with a trainable variable `gamma`, and center
    the outputs by broadcasting with a trainable variable `beta`. `gamma` will
    default to a ones tensor and `beta` will default to a zeros tensor, so that
    centering and scaling are no-ops before training has begun.

    So, with scaling and centering enabled the normalization equations
    are as follows:

    Let the intermediate activations for a mini-batch to be the `inputs`.

    For each sample `x_i` in `inputs` with `k` features, we compute the mean and
    variance of the sample:

    ```python
    mean_i = sum(x_i[j] for j in range(k)) / k
    var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k
    ```

    and then compute a normalized `x_i_normalized`, including a small factor
    `epsilon` for numerical stability.

    ```python
    x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
    ```

    And finally `x_i_normalized ` is linearly transformed by `gamma` and `beta`,
    which are learned parameters:

    ```python
    output_i = x_i_normalized * gamma + beta
    ```

    `gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
    this part of the inputs' shape must be fully defined.

    For example:

    >>> layer = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])
    >>> layer.build([5, 20, 30, 40])
    >>> print(layer.beta.shape)
    (20, 30, 40)
    >>> print(layer.gamma.shape)
    (20, 30, 40)

    Note that other implementations of layer normalization may choose to define
    `gamma` and `beta` over a separate set of axes from the axes being
    normalized across. For example, Group Normalization
    ([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
    corresponds to a Layer Normalization that normalizes across height, width,
    and channel and has `gamma` and `beta` span only the channel dimension.
    So, this Layer Normalization implementation will not match a Group
    Normalization layer with group size set to 1.

    Args:
      axis: Integer or List/Tuple. The axis or axes to normalize across.
        Typically, this is the features axis/axes. The left-out axes are
        typically the batch axis/axes. `-1` is the last dimension in the
        input. Defaults to `-1`.
      epsilon: Small float added to variance to avoid dividing by zero. Defaults
        to 1e-3
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored. Defaults to `True`.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling will be done by the next layer.
        Defaults to `True`.
      beta_initializer: Initializer for the beta weight. Defaults to zeros.
      gamma_initializer: Initializer for the gamma weight. Defaults to ones.
      beta_regularizer: Optional regularizer for the beta weight. None by
        default.
      gamma_regularizer: Optional regularizer for the gamma weight. None by
        default.
      beta_constraint: Optional constraint for the beta weight. None by default.
      gamma_constraint: Optional constraint for the gamma weight. None by
        default.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape` (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.

    Output shape:
      Same shape as input.

    Reference:
      - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).
    """

    @utils.allow_initializer_layout
    def __init__(self, axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError("Expected an int or a list/tuple of ints for the argument 'axis', but received: %r" % axis)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.supports_masking = True
        self._fused = None

    def _fused_can_be_used(self, ndims):
        """Returns false if fused implementation cannot be used.

        Check if the axis is contiguous and can be collapsed into the last axis.
        The self.axis is assumed to have no duplicates.
        """
        axis = sorted(self.axis)
        can_use_fused = False
        if axis[-1] == ndims - 1 and axis[-1] - axis[0] == len(axis) - 1:
            can_use_fused = True
        if self.epsilon < 1.001e-05 or self.dtype != 'float32':
            can_use_fused = False
        return can_use_fused

    def build(self, input_shape):
        self.axis = tf_utils.validate_axis(self.axis, input_shape)
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.rank
        param_shape = [input_shape[dim] for dim in self.axis]
        if self.scale:
            self.gamma = self.add_weight(name='gamma', shape=param_shape, initializer=self.gamma_initializer, regularizer=self.gamma_regularizer, constraint=self.gamma_constraint, trainable=True, experimental_autocast=False)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(name='beta', shape=param_shape, initializer=self.beta_initializer, regularizer=self.beta_regularizer, constraint=self.beta_constraint, trainable=True, experimental_autocast=False)
        else:
            self.beta = None
        self._fused = self._fused_can_be_used(rank)
        self.built = True

    def call(self, inputs):
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            inputs_lengths = inputs.nested_row_lengths()
            inputs = inputs.to_tensor()
        inputs = tf.cast(inputs, self.compute_dtype)
        input_shape = inputs.shape
        ndims = len(input_shape)
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape.dims[dim].value

        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and (self.axis != [ndims - 1]):
                return tf.reshape(v, broadcast_shape)
            return v
        if not self._fused:
            input_dtype = inputs.dtype
            if input_dtype in ('float16', 'bfloat16') and self.dtype == 'float32':
                inputs = tf.cast(inputs, 'float32')
            mean, variance = tf.nn.moments(inputs, self.axis, keepdims=True)
            scale, offset = (_broadcast(self.gamma), _broadcast(self.beta))
            outputs = tf.nn.batch_normalization(inputs, mean, variance, offset=offset, scale=scale, variance_epsilon=self.epsilon)
            outputs = tf.cast(outputs, input_dtype)
        else:
            axis = sorted(self.axis)
            tensor_shape = tf.shape(inputs)
            pre_dim = tf.reduce_prod(tensor_shape[:axis[0]])
            in_dim = tf.reduce_prod(tensor_shape[axis[0]:])
            squeezed_shape = [1, pre_dim, in_dim, 1]
            data_format = 'NCHW'
            inputs = tf.reshape(inputs, squeezed_shape)
            scale = tf.ones([pre_dim], dtype=self.dtype)
            offset = tf.zeros([pre_dim], dtype=self.dtype)
            outputs, _, _ = tf.compat.v1.nn.fused_batch_norm(inputs, scale=scale, offset=offset, epsilon=self.epsilon, data_format=data_format)
            outputs = tf.reshape(outputs, tensor_shape)
            scale, offset = (_broadcast(self.gamma), _broadcast(self.beta))
            if scale is not None:
                outputs = outputs * tf.cast(scale, outputs.dtype)
            if offset is not None:
                outputs = outputs + tf.cast(offset, outputs.dtype)
        outputs.set_shape(input_shape)
        if is_ragged:
            outputs = tf.RaggedTensor.from_tensor(outputs, inputs_lengths)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'axis': self.axis, 'epsilon': self.epsilon, 'center': self.center, 'scale': self.scale, 'beta_initializer': initializers.serialize(self.beta_initializer), 'gamma_initializer': initializers.serialize(self.gamma_initializer), 'beta_regularizer': regularizers.serialize(self.beta_regularizer), 'gamma_regularizer': regularizers.serialize(self.gamma_regularizer), 'beta_constraint': constraints.serialize(self.beta_constraint), 'gamma_constraint': constraints.serialize(self.gamma_constraint)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))