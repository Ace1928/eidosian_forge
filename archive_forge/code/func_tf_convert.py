import functools
import importlib
import inspect
import os
import sys
import textwrap
import traceback
from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import directives
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.core import unsupported_features_checker
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import error_utils
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.autograph.tf_convert', v1=[])
def tf_convert(f, ctx, convert_by_default=True, user_requested=False):
    """Decorator that applies AutoGraph to a function.

  Use in internal APIs.

  This API is suitable for high order functions internal to the TensorFlow API,
  and more generally any function to which AutoGraph is not applied.

  Guidance: `convert` was a decorator meant for use directly by developers, but
  most of today's uses go through `tf.function`. `tf_convert` is to be called
  from high order functions internal to TF. By default, all the internal
  TensorFlow functions are skipped when AutoGraph processes the code. This may
  lead to user-supplied functions to be incorrectly skipped as well.
  `tf_convert` helps avoid that. See the following example for more details.

  ```
  =====tf_internal_module.py=====

  def unconverted(input_fn):
    return input_fn()

  def converted(input_fn):
    return tf.__internal__.autograph.tf_convert(
       input_fn, ctx=tf.__internal__.autograph.control_status_ctx())()

  ======user_module.py======

  @tf.function
  def foo(input_fn)
    return unconverted(input_fn)

  @tf.function
  def bar(input_fn)
    return converted(input_fn)

  @tf.function(autograph=False)
  def baz(input_fn)
    return converted(input_fn)
  ```

  The `foo` method above will execute the `input_fn` without autograph
  conversion, while the `bar` method will run an autographed `input_fn`. The
  `baz` method will run an unconverted `input_fn`, since `tf_convert` respect
  the control status context.

  Note that both methods in `tf_internal_module` are skipped by autograph when
  tracing the `tf.function`. The configuration of whether a module/package
  should be skipped by autograph is controlled in
  tensorflow/python/autograph/core/config.py.

  Args:
    f: Callable.
    ctx: ag_ctx.ControlStatusCtx, the Autograph context in which `f` is used.
    convert_by_default: bool, whether to use AutoGraph when the context doesn't
      specify.
    user_requested: bool, whether to ignore the conversion allowlist. See
      ConversionOptions.user_requested.

  Returns:
    Either `f or the converted version of `f`.
  """
    if is_autograph_artifact(f):
        return f
    f_wrapper = f
    decorators, f = tf_decorator.unwrap(f)
    if ctx.status == ag_ctx.Status.ENABLED:
        wrapper_factory = convert(recursive=True, user_requested=user_requested, conversion_ctx=ctx)
    elif ctx.status == ag_ctx.Status.DISABLED:
        wrapper_factory = do_not_convert
    elif ctx.status == ag_ctx.Status.UNSPECIFIED:
        if convert_by_default:
            wrapper_factory = convert(recursive=True, user_requested=user_requested, conversion_ctx=ctx)
        else:
            wrapper_factory = call_with_unspecified_conversion_status
    else:
        assert False, 'This switch contains all possible cases!'
    wrapper = wrapper_factory(f)
    if decorators:
        wrapper = tf_decorator.rewrap(f_wrapper, f, wrapper)
    return autograph_artifact(wrapper)