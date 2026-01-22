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
def GenerateFlag(field, attributes, fix_bools=True, category=None):
    """Generates a flag for a single field in a message.

  Args:
    field: The apitools field object.
    attributes: yaml_arg_schema.Argument, The attributes to use to
      generate the arg.
    fix_bools: True to generate boolean flags as switches that take a value or
      False to just generate them as regular string flags.
    category: The help category to put the flag in.

  Raises:
    ArgumentGenerationError: When an argument could not be generated from the
      API field.

  Returns:
    calliope.base.Argument, The generated argument.
  """
    flag_type, action = GenerateFlagType(field, attributes, fix_bools)
    if isinstance(flag_type, arg_parsers.ArgList):
        choices = None
    else:
        choices = GenerateChoices(field, attributes)
    if field and (not flag_type) and (not action) and (not attributes.processor):
        raise ArgumentGenerationError(field.name, 'The field is of an unknown type. You can specify a type function or a processor to manually handle this argument.')
    name = attributes.arg_name
    arg = base.Argument(name if attributes.is_positional else '--' + name, category=category if not attributes.is_positional else None, action=action or 'store', completer=attributes.completer, help=attributes.help_text, hidden=attributes.hidden)
    if attributes.default != UNSPECIFIED:
        arg.kwargs['default'] = attributes.default
    if action != 'store_true':
        metavar = GetMetavar(attributes.metavar, flag_type, name)
        if metavar:
            arg.kwargs['metavar'] = metavar
        arg.kwargs['type'] = flag_type
        arg.kwargs['choices'] = choices
    if not attributes.is_positional:
        arg.kwargs['required'] = attributes.required
    return arg