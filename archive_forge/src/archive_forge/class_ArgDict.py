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
class ArgDict(arg_utils.RepeatedMessageBindableType):
    """A wrapper to bind an ArgDict argument to a message.

  The non-flat mode has one dict per message. When the field is repeated, you
  can repeat the message by repeating the flag. For example, given a message
  with fields foo and bar, it looks like:

  --arg foo=1,bar=2 --arg foo=3,bar=4

  The Action method below is used later during argument generation to tell
  argparse to allow repeats of the dictionary and to append them.
  """

    @classmethod
    def FromData(cls, data):
        api_field = data['api_field']
        arg_name = data.get('arg_name')
        arg_type = data['type'][ARG_DICT]
        fields = [ArgDictFieldSpec.FromData(d) for d in arg_type['spec']]
        if arg_type.get('flatten'):
            if len(fields) != 2:
                raise InvalidSchemaError('Flattened ArgDicts must have exactly two items in the spec.')
            return FlattenedArgDict(api_field=api_field, arg_name=arg_name, key_spec=fields[0], value_spec=fields[1])
        return cls(api_field, arg_name, fields)

    def __init__(self, api_field, arg_name, fields):
        self.api_field = api_field
        self.arg_name = arg_name
        self.fields = fields

    def Action(self):
        return 'append'

    def GenerateType(self, field):
        """Generates an argparse type function to use to parse the argument.

    The return of the type function will be an instance of the given message
    with the fields filled in.

    Args:
      field: apitools field instance we are generating ArgObject for

    Raises:
      InvalidSchemaError: If a type for a field could not be determined.

    Returns:
      _MessageFieldType, The type function that parses the ArgDict and returns
      a message instance.
    """
        field_spec = _FieldSpec.FromUserData(field, arg_name=self.arg_name, api_field=self.api_field)
        field_specs = [_GetArgDictFieldType(field.type, f) for f in self.fields]
        required = [f.arg_name for f in field_specs if f.required]
        arg_dict = arg_parsers.ArgDict(spec={f.arg_name: f for f in field_specs}, required_keys=required)
        return _MessageFieldType(arg_type=arg_dict, field_spec=field_spec, field_specs=field_specs)