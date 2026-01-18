import tensorboard.lazy as _lazy
@_lazy.lazy_load('tensorboard.compat.tf2')
def tf2():
    """Provide the root module of a TF-2.0 API for use within TensorBoard.

    Returns:
      The root module of a TF-2.0 API, if available.

    Raises:
      ImportError: if a TF-2.0 API is not available.
    """
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v2'):
        return tf.compat.v2
    raise ImportError('cannot import tensorflow 2.0 API')