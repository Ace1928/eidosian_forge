import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def update_docstrings_with_api_lists():
    """Updates the docstrings of dispatch decorators with API lists.

  Updates docstrings for `dispatch_for_api`,
  `dispatch_for_unary_elementwise_apis`, and
  `dispatch_for_binary_elementwise_apis`, by replacing the string '<<API_LIST>>'
  with a list of APIs that have been registered for that decorator.
  """
    _update_docstring_with_api_list(dispatch_for_unary_elementwise_apis, _UNARY_ELEMENTWISE_APIS)
    _update_docstring_with_api_list(dispatch_for_binary_elementwise_apis, _BINARY_ELEMENTWISE_APIS)
    _update_docstring_with_api_list(dispatch_for_binary_elementwise_assert_apis, _BINARY_ELEMENTWISE_ASSERT_APIS)
    _update_docstring_with_api_list(dispatch_for_api, _TYPE_BASED_DISPATCH_SIGNATURES)