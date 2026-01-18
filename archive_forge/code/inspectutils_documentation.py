from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
Constructs a FullArgSpec with each provided attribute, or the default.

    Args:
      args: A list of the argument names accepted by the function.
      varargs: The name of the *varargs argument or None if there isn't one.
      varkw: The name of the **kwargs argument or None if there isn't one.
      defaults: A tuple of the defaults for the arguments that accept defaults.
      kwonlyargs: A list of argument names that must be passed with a keyword.
      kwonlydefaults: A dictionary of keyword only arguments and their defaults.
      annotations: A dictionary of arguments and their annotated types.
    