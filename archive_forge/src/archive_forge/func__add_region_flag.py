from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.transfer.appliances import regions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _add_region_flag(parser, verb):
    """Add region flag for appliances/orders.

  Normally we'd rely on the argument output by region_attribute_config() but
  we can set "choices" and convert the value to lower if we add it this way.

  Args:
    parser (arg_parse.Parser): The parser for the command.
    verb (ResourceVerb): The action taken on the resource, such as 'update'.
  """
    parser.add_argument('--region', choices=regions.CLOUD_REGIONS, type=str.lower, help='The location affiliated with the appliance order to {}.'.format(verb.value))