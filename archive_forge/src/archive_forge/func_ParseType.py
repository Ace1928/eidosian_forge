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
def ParseType(data):
    """Parse the action out of the argument spec.

  Args:
    data: dict, raw arugment data

  Raises:
    ValueError: If the spec is invalid.
    InvalidSchemaError: If spec and non arg_object type are provided.

  Returns:
    The type to use as argparse accepts it.
  """
    contains_spec = 'spec' in data
    if (specified_type := data.get('type')):
        arg_type = specified_type
    elif contains_spec:
        arg_type = ARG_OBJECT
    else:
        arg_type = None
    if contains_spec and arg_type != ARG_OBJECT:
        arg_name = data.get('arg_name')
        raise InvalidSchemaError(f'Only flags with type arg_object may contain a spec declaration. Flag {arg_name} has type {arg_type}. Update the type or remove the spec declaration.')
    if not arg_type and (not contains_spec):
        return None
    elif isinstance(arg_type, dict) and ARG_DICT in arg_type:
        return ArgDict.FromData(data)
    elif isinstance(arg_type, str):
        return _ParseTypeFromStr(arg_type, data)
    raise ValueError('Unknown value for type: ' + str(arg_type))