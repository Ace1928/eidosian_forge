import math
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import random
from keras.src.initializers.initializer import Initializer
from keras.src.saving import serialization_lib
@keras_export(['keras.initializers.OrthogonalInitializer', 'keras.initializers.Orthogonal', 'keras.initializers.orthogonal'])
class OrthogonalInitializer(Initializer):
    """Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Examples:

    >>> # Standalone usage:
    >>> initializer = keras.initializers.Orthogonal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = keras.initializers.Orthogonal()
    >>> layer = keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to make the behavior of the initializer
            deterministic.

    Reference:

    - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
    """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self._init_seed = seed
        self.seed = seed or random.make_default_seed()

    def __call__(self, shape, dtype=None):
        if len(shape) < 2:
            raise ValueError(f'The tensor to initialize must be at least two-dimensional. Received: shape={shape} of rank {len(shape)}.')
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))
        a = random.normal(flat_shape, seed=self.seed, dtype=dtype)
        q, r = ops.qr(a)
        d = ops.diag(r)
        q *= ops.sign(d)
        if num_rows < num_cols:
            q = ops.transpose(q)
        return self.gain * ops.reshape(q, shape)

    def get_config(self):
        seed_config = serialization_lib.serialize_keras_object(self._init_seed)
        return {'gain': self.gain, 'seed': seed_config}