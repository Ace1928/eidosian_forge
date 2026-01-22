import inspect
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
def allow_initializer_layout(init_method):
    """A decorator for injecting layout information to layer.__init__.

    Layout will be a new param for any of the weights for all the keras layers.
    Adding the param to all the __init__ method will be a big/duplicated work.

    This decorator is design to reduce and code duplication and make it easy to
    add/remove the dtensor feature if needed.

    Sample usage:
    ```python
    class Dense(tf.keras.layer.Layer):

      @allow_initializer_layout
      def __init__(self, units,
                   kernel_initializer='zeros',
                   bias_initializer='zeros',
                   **kwargs):
         super().__init__(**kwargs)

    d = Dense(units=8, kernel_layout=layout1, bias_layout=layout2)
    d.kernel_layout == layout1
    d.bias_layout == layout2
    ```

    By adding this annotation, it will:

    1. Filter out the kwargs based on some keywords, eg if the
      'kernel_initialzer' appears in method signature, then it will try to pop
      the 'kernel_layout' if it presents. Same for "bias" and
      "recurrent_kernel", etc. This will make sure the layout related param is
      not passed to `BaseLayer.__init__`, which will raise error about unexpect
      keyword args.
    2. Set the self.kernel/bias_layout attribute after the `__init__` method is
       called. Keras framework will use those fields to create weights down the
       stream.

    Args:
      init_method: the `__init__` method of the Keras layer to annotate.

    Returns:
      the annotated __init__ method.
    """

    def _wrap_function(layer_instance, *args, **kwargs):
        signature = inspect.signature(init_method)
        layout_args = {}
        for variable_name in KERAS_VARIABLE_NAMES:
            if variable_name + '_initializer' in signature.parameters:
                layout = kwargs.pop(variable_name + '_layout', None)
                if layout:
                    layout_args[variable_name + '_layout'] = layout
        init_method(layer_instance, *args, **kwargs)
        for layout_param_name, layout in layout_args.items():
            setattr(layer_instance, layout_param_name, layout)
    return tf.__internal__.decorator.make_decorator(target=init_method, decorator_func=_wrap_function)