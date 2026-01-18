import tensorboard.lazy as _lazy
@_lazy.lazy_load('tensorboard.compat.tf')
def tf():
    """Provide the root module of a TF-like API for use within TensorBoard.

    By default this is equivalent to `import tensorflow as tf`, but it can be used
    in combination with //tensorboard/compat:tensorflow (to fall back to a stub TF
    API implementation if the real one is not available) or with
    //tensorboard/compat:no_tensorflow (to force unconditional use of the stub).

    Returns:
      The root module of a TF-like API, if available.

    Raises:
      ImportError: if a TF-like API is not available.
    """
    try:
        from tensorboard.compat import notf
    except ImportError:
        try:
            import tensorflow
            return tensorflow
        except ImportError:
            pass
    from tensorboard.compat import tensorflow_stub
    return tensorflow_stub