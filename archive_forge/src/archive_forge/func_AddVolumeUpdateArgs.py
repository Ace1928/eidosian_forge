from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeUpdateArgs(parser, release_track):
    """Add args for updating a Volume."""
    concept_parsers.ConceptParser([flags.GetVolumePresentationSpec('The Volume to update.')]).AddToParser(parser)
    messages = netapp_api_util.GetMessagesModule(release_track=release_track)
    flags.AddResourceDescriptionArg(parser, 'Volume')
    flags.AddResourceCapacityArg(parser, 'Volume', required=False)
    AddVolumeAssociatedStoragePoolArg(parser, required=False)
    flags.AddResourceAsyncFlag(parser)
    AddVolumeProtocolsArg(parser, required=False)
    AddVolumeShareNameArg(parser, required=False)
    AddVolumeExportPolicyArg(parser)
    AddVolumeUnixPermissionsArg(parser)
    AddVolumeSmbSettingsArg(parser)
    AddVolumeSourceSnapshotArg(parser)
    AddVolumeHourlySnapshotArg(parser)
    AddVolumeDailySnapshotArg(parser)
    AddVolumeWeeklySnapshotArg(parser)
    AddVolumeMonthlySnapshotArg(parser)
    AddVolumeSnapReserveArg(parser)
    AddVolumeSnapshotDirectoryArg(parser)
    AddVolumeSecurityStyleArg(parser, messages)
    AddVolumeEnableKerberosArg(parser)
    AddVolumeRestrictedActionsArg(parser)
    if release_track == calliope_base.ReleaseTrack.BETA or release_track == calliope_base.ReleaseTrack.GA:
        AddVolumeBackupConfigArg(parser)
        AddVolumeSourceBackupArg(parser)
    if release_track == calliope_base.ReleaseTrack.ALPHA or release_track == calliope_base.ReleaseTrack.BETA:
        AddVolumeTieringPolicyArg(parser, messages)
    labels_util.AddUpdateLabelsFlags(parser)