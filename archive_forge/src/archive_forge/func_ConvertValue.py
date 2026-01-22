from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
def ConvertValue(field, value, repeated=None, processor=None, choices=None):
    """Coverts the parsed value into something to insert into a request message.

  If a processor is registered, that is called on the value.
  If a choices mapping was provided, each value is mapped back into its original
  value.
  If the field is an enum, the value will be looked up by name and the Enum type
  constructed.

  Args:
    field: The apitools field object.
    value: The parsed value. This must be a scalar for scalar fields and a list
      for repeated fields.
    repeated: bool, Set to False if this arg was forced to be singular even
      though the API field it corresponds to is repeated.
    processor: A function to process the value before putting it into the
      message.
    choices: {str: str} A mapping of argument value, to enum API enum value.

  Returns:
    The value to insert into the message.
  """
    arg_repeated = field.repeated and repeated is not False
    if processor:
        value = processor(value)
    else:
        valid_choices = None
        if choices:
            valid_choices = choices.keys()
            if field.variant == messages.Variant.ENUM:
                api_names = field.type.names()
            else:
                api_names = []
            CheckValidEnumNames(api_names, choices.values())
            if arg_repeated:
                value = [_MapChoice(choices, v) for v in value]
            else:
                value = _MapChoice(choices, value)
        if field.variant == messages.Variant.ENUM:
            t = field.type
            if arg_repeated:
                value = [ChoiceToEnum(v, t, valid_choices=valid_choices) for v in value]
            else:
                value = ChoiceToEnum(value, t, valid_choices=valid_choices)
    if field.repeated and (not arg_repeated) and (not isinstance(value, list)):
        value = [value]
    return value