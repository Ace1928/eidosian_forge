from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import json
import os
import pipes
import re
import shlex
import sys
import types
from fire import completion
from fire import decorators
from fire import formatting
from fire import helptext
from fire import inspectutils
from fire import interact
from fire import parser
from fire import trace
from fire import value_types
from fire.console import console_io
import six
def _ParseKeywordArgs(args, fn_spec):
    """Parses the supplied arguments for keyword arguments.

  Given a list of arguments, finds occurrences of --name value, and uses 'name'
  as the keyword and 'value' as the value. Constructs and returns a dictionary
  of these keyword arguments, and returns a list of the remaining arguments.

  Only if fn_keywords is None, this only finds argument names used by the
  function, specified through fn_args.

  This returns the values of the args as strings. They are later processed by
  _ParseArgs, which converts them to the appropriate type.

  Args:
    args: A list of arguments.
    fn_spec: The inspectutils.FullArgSpec describing the given callable.
  Returns:
    kwargs: A dictionary mapping keywords to values.
    remaining_kwargs: A list of the unused kwargs from the original args.
    remaining_args: A list of the unused arguments from the original args.
  Raises:
    FireError: If a single-character flag is passed that could refer to multiple
        possible args.
  """
    kwargs = {}
    remaining_kwargs = []
    remaining_args = []
    fn_keywords = fn_spec.varkw
    fn_args = fn_spec.args + fn_spec.kwonlyargs
    if not args:
        return (kwargs, remaining_kwargs, remaining_args)
    skip_argument = False
    for index, argument in enumerate(args):
        if skip_argument:
            skip_argument = False
            continue
        if _IsFlag(argument):
            contains_equals = '=' in argument
            stripped_argument = argument.lstrip('-')
            if contains_equals:
                key, value = stripped_argument.split('=', 1)
            else:
                key = stripped_argument
            key = key.replace('-', '_')
            is_bool_syntax = not contains_equals and (index + 1 == len(args) or _IsFlag(args[index + 1]))
            keyword = ''
            if key in fn_args or (is_bool_syntax and key.startswith('no') and (key[2:] in fn_args)) or fn_keywords:
                keyword = key
            elif len(key) == 1:
                matching_fn_args = [arg for arg in fn_args if arg[0] == key]
                if len(matching_fn_args) == 1:
                    keyword = matching_fn_args[0]
                elif len(matching_fn_args) > 1:
                    raise FireError("The argument '{}' is ambiguous as it could refer to any of the following arguments: {}".format(argument, matching_fn_args))
            if not keyword:
                got_argument = False
            elif contains_equals:
                got_argument = True
            elif is_bool_syntax:
                got_argument = True
                if keyword in fn_args:
                    value = 'True'
                elif keyword.startswith('no'):
                    keyword = keyword[2:]
                    value = 'False'
                else:
                    value = 'True'
            else:
                assert index + 1 < len(args)
                value = args[index + 1]
                got_argument = True
            skip_argument = not contains_equals and (not is_bool_syntax)
            if got_argument:
                kwargs[keyword] = value
            else:
                remaining_kwargs.append(argument)
                if skip_argument:
                    remaining_kwargs.append(args[index + 1])
        else:
            remaining_args.append(argument)
    return (kwargs, remaining_kwargs, remaining_args)