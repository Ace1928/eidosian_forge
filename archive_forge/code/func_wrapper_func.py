import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
def wrapper_func(tuple_path, *inputs, **kwargs):
    string_path = '/'.join((str(s) for s in tuple_path))
    return func(string_path, *inputs, **kwargs)