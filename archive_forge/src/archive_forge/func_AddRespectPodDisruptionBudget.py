from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddRespectPodDisruptionBudget(parser):
    """Adds --respect-pdb flag to parser."""
    help_text = 'Indicates whether the node pool rollback should respect pod disruption budget.\n'
    parser.add_argument('--respect-pdb', default=False, action='store_true', help=help_text)