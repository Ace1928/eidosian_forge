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
class _FieldSpecType(usage_text.DefaultArgTypeWrapper, metaclass=abc.ABCMeta):
    """Wrapper that holds the arg type and information about the type.

  Interface allows users to parse string into arg_type and then parse value
  into correct apitools field.

  Attributes:
    field: apitools field instance
    api_field: str, name of the field where value should be mapped in message.
    arg_name: str, name of key in dict.
    repeated: bool, whether the field is repeated.
    required: bool, whether the field value is required.
  """

    def __init__(self, arg_type, field_spec):
        super(_FieldSpecType, self).__init__(arg_type=arg_type)
        self.field = field_spec.field
        self.api_field = field_spec.api_field
        self.arg_name = field_spec.arg_name
        self.repeated = field_spec.repeated
        self.required = field_spec.required

    def ParseIntoMessage(self, message_instance, value):
        """Sets field in a message after value is parsed into correct type.

    Args:
      message_instance: apitools message instance we are parsing value into
      value: value we are parsing into apitools message
    """
        if value is None and self.repeated:
            field_value = []
        else:
            field_value = value
        arg_utils.SetFieldInMessage(message_instance, self.api_field, field_value)

    @abc.abstractmethod
    def __call__(self, arg_value):
        """Parses arg_value into apitools message using field specs provided."""