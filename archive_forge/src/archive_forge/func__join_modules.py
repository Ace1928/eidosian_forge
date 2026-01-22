import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
def _join_modules(module1, module2):
    """Concatenate 2 module components.

  Args:
    module1: First module to join.
    module2: Second module to join.

  Returns:
    Given two modules aaa.bbb and ccc.ddd, returns a joined
    module aaa.bbb.ccc.ddd.
  """
    if not module1:
        return module2
    if not module2:
        return module1
    return '%s.%s' % (module1, module2)