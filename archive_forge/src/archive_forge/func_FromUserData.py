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
@classmethod
def FromUserData(cls, field, api_field=None, arg_name=None, required=None, hidden=False):
    """Creates a _FieldSpec from user input.

    If value is not provided in yaml schema by user, the value is defaulted
    to a value derived from the apitools field.

    Args:
      field: apitools field instance
      api_field: The name of the field under the repeated message that the value
        should be put.
      arg_name: The name of the key in the dict.
      required: True if the key is required.
      hidden: True if the help text should be hidden.

    Returns:
      _FieldSpec instance

    Raises:
      ValueError: if the field contradicts the values provided by the user
    """
    field_name = api_field or field.name
    child_field_name = arg_utils.GetChildFieldName(field_name)
    if child_field_name != field.name:
        raise ValueError(f'Expected to receive field {child_field_name} but got {field.name}')
    return cls(field=field, api_field=field_name, arg_name=arg_name or child_field_name, repeated=field.repeated, required=required if required is not None else field.required, hidden=hidden)