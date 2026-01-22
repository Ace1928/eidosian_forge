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
class FlattenedArgDict(arg_utils.RepeatedMessageBindableType):
    """A wrapper to bind an ArgDict argument to a message with a key/value pair.

  The flat mode has one dict corresponding to a repeated field. For example,
  given a message with fields key and value, it looks like:

  --arg a=b,c=d

  Which would generate 2 instances of the message:
  [{key=a, value=b}, {key=c, value=d}]
  """

    def __init__(self, api_field, arg_name, key_spec, value_spec):
        self.api_field = api_field
        self.arg_name = arg_name
        self.key_spec = key_spec
        self.value_spec = value_spec

    def GenerateType(self, field):
        """Generates an argparse type function to use to parse the argument.

    The return of the type function will be a list of instances of the given
    message with the fields filled in.

    Args:
      field: apitools field instance we are generating ArgObject for

    Raises:
      InvalidSchemaError: If a type for a field could not be determined.

    Returns:
      _AdditionalPropsType, The type function that parses the ArgDict
        and returns a list of message instances.
    """
        field_spec = _FieldSpec.FromUserData(field, arg_name=self.arg_name, api_field=self.api_field)
        key_type = _GetArgDictFieldType(field.type, self.key_spec)
        value_type = _GetArgDictFieldType(field.type, self.value_spec)
        arg_dict = arg_parsers.ArgDict(key_type=key_type, value_type=value_type)
        return _AdditionalPropsType(arg_type=arg_dict, field_spec=field_spec, key_spec=key_type, value_spec=value_type)