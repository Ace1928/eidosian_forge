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
def _GenerateFieldType(self, field_spec, is_label_field=False):
    """Returns _FieldType that parses apitools field from string.

    Args:
      field_spec: _FieldSpec, information about the field
      is_label_field: bool, whether or not the field is for a labels map field.
        If true, supplies default validation and help text.

    Returns:
      _FieldType that takes string like '1' or ['1'] and parses it
      into 1 or [1] depending on the apitools field type
    """
    if is_label_field and field_spec.arg_name == KEY:
        value_type = labels_util.KEY_FORMAT_VALIDATOR
        default_help_text = labels_util.KEY_FORMAT_HELP
    elif is_label_field and field_spec.arg_name == VALUE:
        value_type = labels_util.VALUE_FORMAT_VALIDATOR
        default_help_text = labels_util.VALUE_FORMAT_HELP
    else:
        value_type = _GetFieldValueType(field_spec.field)
        default_help_text = None
    arg_obj = arg_parsers.ArgObject(value_type=value_type, help_text=self.help_text or default_help_text, repeated=field_spec.repeated, hidden=field_spec.hidden, enable_shorthand=False)
    return _FieldType(arg_type=arg_obj, field_spec=field_spec, choices=None)