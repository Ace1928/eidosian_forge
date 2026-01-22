from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _ValidateAndParseColors(value):
    """Validates that values has proper format and returns parsed components."""
    values = value.split(',')
    if len(values) == 3:
        try:
            return [_ConvertColorValue(x) for x in values]
        except ValueError:
            raise RedactColorError('Invalid Color Value(s) [{}]. {}'.format(value, _COLOR_SPEC_ERROR_SUFFIX))
    else:
        raise RedactColorError('You must specify exactly 3 color values [{}]. {}'.format(value, _COLOR_SPEC_ERROR_SUFFIX))