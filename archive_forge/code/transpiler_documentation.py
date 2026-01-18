import inspect
import threading
import types
import gast
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
Transforms a function. See GenericTranspiler.trasnform_function.

    This overload wraps the parent's `transform_function`, adding caching and
    facilities to instantiate the output as a Python object. It also
    adds facilities to make new symbols available to the generated Python code,
    visible as local variables - see `get_extra_locals`.

    Args:
      fn: A function or lambda.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user attribute.
    Returns:
      A tuple:
        * A function or lambda with the same signature and closure as `fn`
        * The temporary module into which the transformed function was loaded
        * The source map as a
            Dict[origin_info.LineLocation, origin_info.OriginInfo]
    