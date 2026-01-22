from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp.volumes import client as volumes_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp.volumes import flags as volumes_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Revert(base.Command):
    """Revert a Cloud NetApp Volume back to a specified Snapshot."""
    _RELEASE_TRACK = base.ReleaseTrack.GA
    detailed_help = {'DESCRIPTION': '          Revert a Cloud NetApp Volume back to a specified source Snapshot\n          ', 'EXAMPLES': '          The following command reverts a Volume named NAME in the given location and snapshot\n\n              $ {command} NAME --location=us-central1 --snapshot="snapshot1"\n          '}

    @staticmethod
    def Args(parser):
        concept_parsers.ConceptParser([flags.GetVolumePresentationSpec('The Volume to revert.')]).AddToParser(parser)
        volumes_flags.AddVolumeRevertSnapshotArg(parser)
        flags.AddResourceAsyncFlag(parser)

    def Run(self, args):
        """Run the revert command."""
        volume_ref = args.CONCEPTS.volume.Parse()
        client = volumes_client.VolumesClient(release_track=self._RELEASE_TRACK)
        revert_warning = 'You are about to revert Volume {} back to Snapshot {}.\nAre you sure?'.format(volume_ref.RelativeName(), args.snapshot)
        if not console_io.PromptContinue(message=revert_warning):
            return None
        result = client.RevertVolume(volume_ref, args.snapshot, args.async_)
        if args.async_:
            command = 'gcloud {} netapp volumes list'.format(self.ReleaseTrack().prefix)
            log.status.Print('Check the status of the volume being reverted by listing all volumes:\n$ {}'.format(command))
        return result