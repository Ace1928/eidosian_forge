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
def GenerateFlagType(field, attributes, fix_bools=True):
    """Generates the type and action for a flag.

  Translates the yaml type (or deault apitools type) to python type. If the
  type is for a repeated field, then a function that turns the input into an
  apitools message is returned.

  Args:
    field: apitools field object flag is associated with
    attributes: yaml_arg_schema.Argument, data about flag being generated
    fix_bools: bool, whether to update flags to store_true action

  Raises:
    ArgumentGenerationError: user cannot specify action for repeated field
    ArgumentGenerationError: cannot use a dictionary on a non-repeating field
    ArgumentGenerationError: append action can only be used for repeated fields

  Returns:
    (str) -> Any, a type or function that returns input into correct type
    action, flag action used with a given type
  """
    variant = field.variant if field else None
    flag_type = attributes.type or TYPES.get(variant, None)
    action = attributes.action
    if flag_type == bool and fix_bools and (not action):
        action = 'store_true'
    append_action = 'append'
    repeated = (field and field.repeated) and attributes.repeated is not False
    if isinstance(flag_type, ArgObjectType):
        if action:
            raise ArgumentGenerationError(field.name, 'Type {0} cannot be used with a custom action. Remove action {1} from spec.'.format(type(flag_type).__name__, action))
        action = flag_type.Action(field)
        flag_type = flag_type.GenerateType(field)
    elif repeated:
        if flag_type:
            is_repeatable_message = isinstance(flag_type, RepeatedMessageBindableType)
            is_arg_list = isinstance(flag_type, arg_parsers.ArgList)
            if (is_repeatable_message or is_arg_list) and action:
                raise ArgumentGenerationError(field.name, 'Type {0} cannot be used with a custom action. Remove action {1} from spec.'.format(type(flag_type).__name__, action))
            if is_repeatable_message:
                action = flag_type.Action()
                flag_type = flag_type.GenerateType(field)
            elif not is_arg_list and action != append_action:
                flag_type = arg_parsers.ArgList(element_type=flag_type, choices=GenerateChoices(field, attributes))
    elif isinstance(flag_type, RepeatedMessageBindableType):
        raise ArgumentGenerationError(field.name, 'Type {0} can only be used on repeated fields.'.format(type(flag_type).__name__))
    elif action == append_action:
        raise ArgumentGenerationError(field.name, '{0} custom action can only be used on repeated fields.'.format(action))
    return (flag_type, action)