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
class _FieldType(_FieldSpecType):
    """Type that converts string into apitools field instance.

  Attributes:
    choices: list[Choice], list of valid user inputs
  """

    def __init__(self, choices=None, **kwargs):
        super(_FieldType, self).__init__(**kwargs)
        self.choices = choices

    def __call__(self, arg_value):
        """Converts string into apitools field value."""
        parsed_arg_value = self.arg_type(arg_value)
        return arg_utils.ConvertValue(self.field, parsed_arg_value, repeated=self.repeated, choices=self.choices)