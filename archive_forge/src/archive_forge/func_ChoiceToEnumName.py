from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.identity import admin_directory
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def ChoiceToEnumName(choice):
    """Converts an argument value to the string representation of the Enum."""
    return choice.replace('-', '_')