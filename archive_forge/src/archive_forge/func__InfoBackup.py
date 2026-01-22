from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
def _InfoBackup(component):
    """Returns a dict with information about the given component.

  This function is to be called only in the case that IPython's
  oinspect module is not available. The info dict it produces may
  contain less information that contained in the info dict produced
  by oinspect.

  Args:
    component: The component to analyze.
  Returns:
    A dict with information about the component.
  """
    info = {}
    info['type_name'] = type(component).__name__
    info['string_form'] = str(component)
    filename, lineno = GetFileAndLine(component)
    info['file'] = filename
    info['line'] = lineno
    info['docstring'] = inspect.getdoc(component)
    try:
        info['length'] = str(len(component))
    except (TypeError, AttributeError):
        pass
    return info