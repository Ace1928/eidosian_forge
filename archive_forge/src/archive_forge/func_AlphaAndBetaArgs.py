from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import environment_patch_util as patch_util
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
@staticmethod
def AlphaAndBetaArgs(parser, release_track=base.ReleaseTrack.BETA):
    """Arguments available only in both alpha and beta."""
    Update.Args(parser, release_track=release_track)
    UpdateBeta.support_environment_upgrades = True
    flags.AddEnvUpgradeFlagsToGroup(Update.update_type_group)
    flags.AddComposer3FlagsToGroup(Update.update_type_group)