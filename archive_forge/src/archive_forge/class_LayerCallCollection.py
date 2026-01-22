import functools
import threading
import weakref
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
class LayerCallCollection(object):
    """Groups wrapped layer call functions.

  This is used to ensure that all layer call functions are traced with the same
  inputs-
    - call
    - call_and_return_conditional_losses
    - call_and_return_all_conditional_losses
  """

    def __init__(self, layer):
        self.layer = layer
        self.layer_call_method = _get_layer_call_method(layer)
        self._expects_training_arg = utils.layer_uses_training_bool(layer)
        self._training_arg_index = utils.get_training_arg_index(self.layer_call_method)
        arg_spec = tf_inspect.getfullargspec(self.layer_call_method)
        self._has_kwargs = bool(self._expects_training_arg or arg_spec.defaults or arg_spec.kwonlyargs or arg_spec.varkw)
        self._input_signature = self._generate_input_signature(layer)
        self._functions = weakref.WeakValueDictionary()
        args = arg_spec.args
        if tf_inspect.ismethod(self.layer_call_method):
            args = args[1:]
        self._input_arg_name = args[0] if args else 'inputs'

    def _generate_input_signature(self, layer):
        """Inspects layer object and returns the inferred input signature.

    Args:
      layer: Layer object.

    Returns:
      List of possibly nested TensorSpecs of the layer call function inputs.
      The list does not contain the `training` argument.
    """
        if isinstance(layer.call, def_function.Function) and layer.call.input_signature is not None:
            return layer.call.input_signature
        elif isinstance(layer, training_lib.Model):
            return saving_utils.model_input_signature(layer)
        elif layer.input_spec is not None and layer._use_input_spec_as_call_signature:

            def to_tensor_spec_or_none(x):
                spec = input_spec.to_tensor_spec(x, layer._compute_dtype)
                if spec.shape == tensor_shape.TensorShape(None):
                    return None
                return spec
            input_signature = [nest.map_structure(to_tensor_spec_or_none, layer.input_spec)]
            return input_signature
        else:
            return None

    def add_trace(self, *args, **kwargs):
        """Traces all functions with the same args and kwargs.

    Args:
      *args: Positional args passed to the original function.
      **kwargs: Keyword args passed to the original function.
    """
        args = list(args)
        kwargs = kwargs.copy()
        for fn in self._functions.values():
            if self._expects_training_arg:

                def trace_with_training(value, fn=fn):
                    utils.set_training_arg(value, self._training_arg_index, args, kwargs)
                    add_trace_to_queue(fn, args, kwargs, value)
                trace_with_training(True)
                trace_with_training(False)
            else:
                add_trace_to_queue(fn, args, kwargs)

    @property
    def fn_input_signature(self):
        """Returns input signature for the wrapped layer call function."""
        if self._has_kwargs:
            return None
        if None in nest.flatten(self._input_signature):
            return None
        return self._input_signature

    def training_arg_was_passed(self, args, kwargs):
        if not self.layer._expects_training_arg and self._expects_training_arg:
            return utils.get_training_arg(self._training_arg_index, args, kwargs) is not None
        else:
            return self.layer._call_arg_was_passed('training', args, kwargs, inputs_in_args=True)

    def get_training_arg_value(self, args, kwargs):
        if not self.layer._expects_training_arg and self._expects_training_arg:
            return utils.get_training_arg(self._training_arg_index, args, kwargs)
        else:
            return self.layer._get_call_arg_value('training', args, kwargs, inputs_in_args=True)

    def get_input_arg_value(self, args, kwargs):
        return self.layer._get_call_arg_value(self._input_arg_name, args, kwargs, inputs_in_args=True)

    def _maybe_wrap_with_training_arg(self, call_fn, match_layer_training_arg):
        """Wraps call function with added training argument if necessary."""
        if not self.layer._expects_training_arg and self._expects_training_arg:
            arg_spec = tf_inspect.getfullargspec(call_fn)
            args = arg_spec.args + ['training']
            defaults = list(arg_spec.defaults or [])
            defaults.append(False)
            new_arg_spec = tf_inspect.FullArgSpec(args=args, varargs=arg_spec.varargs, varkw=arg_spec.varkw, defaults=defaults, kwonlyargs=arg_spec.kwonlyargs, kwonlydefaults=arg_spec.kwonlydefaults, annotations=arg_spec.annotations)
            self._training_arg_index = len(args) - 1
            if tf_inspect.ismethod(call_fn):
                self._training_arg_index -= 1

            def wrap_with_training_arg(*args, **kwargs):
                if match_layer_training_arg:
                    args = list(args)
                    kwargs = kwargs.copy()
                    utils.remove_training_arg(self._training_arg_index, args, kwargs)
                return call_fn(*args, **kwargs)
            return tf_decorator.make_decorator(target=call_fn, decorator_func=wrap_with_training_arg, decorator_argspec=new_arg_spec)
        return call_fn

    def add_function(self, call_fn, name, match_layer_training_arg):
        """Adds a layer call function to the collection.

    Args:
      call_fn: a python function
      name: Name of call function
      match_layer_training_arg: If True, removes the `training` from the
        function arguments when calling `call_fn`.

    Returns:
      LayerCall (tf.function)
    """
        fn = LayerCall(self, self._maybe_wrap_with_training_arg(call_fn, match_layer_training_arg), name, input_signature=self.fn_input_signature)
        self._functions[name] = fn.wrapped_call
        return fn

    def trace_with_input_signature(self):
        """Trace with the layer/models inferred input signature if possible."""
        if None not in nest.flatten(self._input_signature) and self._has_kwargs:
            self.add_trace(*self._input_signature)