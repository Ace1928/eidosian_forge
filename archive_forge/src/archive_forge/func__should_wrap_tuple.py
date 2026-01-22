import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _should_wrap_tuple(t):
    """Determine if a tuple has any trackable components."""
    for element in t:
        if isinstance(element, NoDependency):
            return True
        if isinstance(element, base.Trackable):
            return True
        if type(element) == dict:
            return True
        if type(element) == collections.OrderedDict:
            return True
        if type(element) == list:
            return True
        if isinstance(element, tuple) and _should_wrap_tuple(element):
            return True
    return False