from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddEnforcedRetention(parser, required):
    """Adds a positional enforced-retention argument to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    required: Whether or not --enforced-retention is required.
  """
    parser.add_argument('--enforced-retention', required=required, type=arg_parsers.Duration(lower_bound='1d', upper_bound='36159d', parsed_unit='s'), help='Backups will be kept for this minimum period before they can be deleted. Once the effective time is reached, the enforced retention period cannot be decreased or removed. ')