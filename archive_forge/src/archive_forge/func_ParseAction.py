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
def ParseAction(action, flag_name):
    """Parse the action out of the argument spec.

  Args:
    action: The argument action spec data.
    flag_name: str, The effective flag name.

  Raises:
    ValueError: If the spec is invalid.

  Returns:
    The action to use as argparse accepts it. It will either be a class that
    implements action, or it will be a str of a builtin argparse type.
  """
    if not action:
        return None
    if isinstance(action, str):
        if action in STATIC_ACTIONS:
            return action
        return Hook.FromPath(action)
    deprecation = action.get('deprecated')
    if deprecation:
        return actions.DeprecationAction(flag_name, **deprecation)
    raise ValueError('Unknown value for action: ' + str(action))