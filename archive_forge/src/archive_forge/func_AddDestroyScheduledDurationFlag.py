from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def AddDestroyScheduledDurationFlag(parser):
    parser.add_argument('--destroy-scheduled-duration', type=arg_parsers.Duration(upper_bound='120d'), help='The amount of time that versions of the key should spend in the DESTROY_SCHEDULED state before transitioning to DESTROYED. See $ gcloud topic datetimes for information on duration formats.')