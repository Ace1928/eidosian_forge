from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import abc
from collections.abc import Callable
import dataclasses
from typing import Any
from apitools.base.protorpclite import messages as apitools_messages
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import module_util
def ImportPythonHook(path):
    """Imports the given python hook.

  Depending on what it is used for, a hook is a reference to a class, function,
  or attribute in Python code.

  Args:
    path: str, The path of the hook to import. It must be in the form of:
      package.module:attribute.attribute where the module path is separated from
      the class name and sub attributes by a ':'. Additionally, ":arg=value,..."
      can be appended to call the function with the given args and use the
      return value as the hook.

  Raises:
    InvalidSchemaError: If the given module or attribute cannot be loaded.

  Returns:
    Hook, the hook configuration.
  """
    parts = path.split(':')
    if len(parts) != 2 and len(parts) != 3:
        raise InvalidSchemaError('Invalid Python hook: [{}]. Hooks must be in the format: package(.module)+:attribute(.attribute)*(:arg=value(,arg=value)*)?'.format(path))
    try:
        attr = module_util.ImportModule(parts[0] + ':' + parts[1])
    except module_util.ImportModuleError as e:
        raise InvalidSchemaError('Could not import Python hook: [{}]. {}'.format(path, e))
    kwargs = None
    if len(parts) == 3:
        kwargs = {}
        for arg in parts[2].split(','):
            if not arg:
                continue
            arg_parts = arg.split('=')
            if len(arg_parts) != 2:
                raise InvalidSchemaError('Invalid Python hook: [{}]. Args must be in the form arg=value,arg=value,...'.format(path))
            kwargs[arg_parts[0].strip()] = arg_parts[1].strip()
    return Hook(attr, kwargs)