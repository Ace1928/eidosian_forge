from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam.byoid_utilities import cred_config
from googlecloudsdk.command_lib.util.args import common_args
def GetOrgFlag(verb):
    return base.Argument('--organization', help='Organization of the role you want to {0}.'.format(verb))