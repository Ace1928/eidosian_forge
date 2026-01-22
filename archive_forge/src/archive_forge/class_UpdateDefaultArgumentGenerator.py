from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
import six
class UpdateDefaultArgumentGenerator(UpdateBasicArgumentGenerator):
    """Update flag generator for simple values."""

    @property
    def _empty_value(self):
        return None

    @property
    def set_arg(self):
        return self._CreateBasicFlag(flag_type=_ConvertValueType(self), action=self.action, metavar=self.metavar, help_text='Set {} to new value.'.format(self.arg_name))

    @property
    def clear_arg(self):
        return self._CreateBasicFlag(flag_prefix=Prefix.CLEAR, action='store_true', help_text='Clear {} value and set to {}.'.format(self.arg_name, self._GetTextFormatOfEmptyValue(self._empty_value)))

    def ApplySetFlag(self, existing_val, set_val):
        if set_val:
            return set_val
        return existing_val

    def ApplyClearFlag(self, existing_val, clear_flag):
        if clear_flag:
            return self._empty_value
        return existing_val