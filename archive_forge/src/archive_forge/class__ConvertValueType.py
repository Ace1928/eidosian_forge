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
class _ConvertValueType(usage_text.DefaultArgTypeWrapper):
    """Wraps flag types in arg_utils.ConvertValue while maintaining help text.

  Attributes:
    arg_gen: UpdateBasicArgumentGenerator, update argument generator
  """

    def __init__(self, arg_gen):
        super(_ConvertValueType, self).__init__(arg_gen.flag_type)
        self.field = arg_gen.field
        self.repeated = arg_gen.repeated
        self.processor = arg_gen.processor
        self.choices = arg_gen.choices

    def __call__(self, arg_value):
        """Converts arg_value into type arg_type."""
        value = self.arg_type(arg_value)
        return arg_utils.ConvertValue(self.field, value, repeated=self.repeated, processor=self.processor, choices=util.Choice.ToChoiceMap(self.choices))