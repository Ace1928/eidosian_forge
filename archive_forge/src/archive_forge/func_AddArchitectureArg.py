from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def AddArchitectureArg(parser, messages):
    """Add the image architecture arg."""
    architecture_enum_type = messages.Image.ArchitectureValueValuesEnum
    excluded_enums = [architecture_enum_type.ARCHITECTURE_UNSPECIFIED.name]
    architecture_choices = sorted([e for e in architecture_enum_type.names() if e not in excluded_enums])
    parser.add_argument('--architecture', choices=architecture_choices, help='Specifies the architecture or processor type that this image can support. For available processor types on Compute Engine, see https://cloud.google.com/compute/docs/cpu-platforms.')