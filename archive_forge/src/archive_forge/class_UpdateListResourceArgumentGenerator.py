from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
class UpdateListResourceArgumentGenerator(UpdateResourceArgumentGenerator):
    """Update flag generator for list resource args."""

    @property
    def _empty_value(self):
        return []

    @property
    def set_arg(self):
        return self._CreateResourceFlag(group_help='Set {} to new value.'.format(self.arg_name))

    @property
    def clear_arg(self):
        return self._CreateFlag(self.arg_name, flag_prefix=update_args.Prefix.CLEAR, action='store_true', help_text='Clear {} value and set to {}.'.format(self.arg_name, self._GetTextFormatOfEmptyValue(self._empty_value)))

    @property
    def update_arg(self):
        return self._CreateResourceFlag(flag_prefix=update_args.Prefix.ADD, group_help='Add new value to {} list.'.format(self.arg_name))

    @property
    def remove_arg(self):
        return self._CreateResourceFlag(flag_prefix=update_args.Prefix.REMOVE, group_help='Remove value from {} list.'.format(self.arg_name))

    def ApplySetFlag(self, output, set_val):
        if set_val:
            return set_val
        return output

    def ApplyClearFlag(self, output, clear_flag):
        if clear_flag:
            return self._empty_value
        return output

    def ApplyRemoveFlag(self, existing_val, remove_val):
        value = existing_val or self._empty_value
        if remove_val:
            return [x for x in value if x not in remove_val]
        else:
            return value

    def ApplyUpdateFlag(self, existing_val, update_val):
        value = existing_val or self._empty_value
        if update_val:
            return existing_val + [x for x in update_val if x not in value]
        else:
            return value