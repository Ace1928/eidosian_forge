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
def ParseEkmConnectionName(args):
    return resources.REGISTRY.Parse(args.ekm_connection, params={'projectsId': properties.VALUES.core.project.GetOrFail, 'locationsId': args.MakeGetOrRaise('--location')}, collection=EKM_CONNECTION_COLLECTION)