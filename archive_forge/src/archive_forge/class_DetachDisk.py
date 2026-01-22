from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import resource_args
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DetachDisk(base.Command):
    """Detach a disk from an instance."""

    @staticmethod
    def Args(parser):
        """Set up arguments for this command.

    Args:
      parser: An argparse.ArgumentParser.
    """
        parser.add_argument('--disk', help='Name of the disk to detach from the TPU VM.', required=True)
        AddTPUResourceArg(parser, 'to detach disk')

    def Run(self, args):
        if args.zone is None:
            args.zone = properties.VALUES.compute.zone.Get(required=True)
        tpu_name_ref = args.CONCEPTS.tpu.Parse()
        tpu = tpu_utils.TPUNode(self.ReleaseTrack())
        node = tpu.Get(tpu_name_ref.Name(), args.zone)
        if not tpu_utils.IsTPUVMNode(node):
            raise exceptions.BadArgumentException('TPU', 'this command is only available for Cloud TPU VM nodes. To access this node, please see https://cloud.google.com/tpu/docs/creating-deleting-tpus.')
        if not node.dataDisks:
            raise exceptions.BadArgumentException('TPU', 'no data disks to detach from current TPU VM.')
        source_path_match = re.match('projects/.+/(zones|regions)/.+/disks/.+', args.disk)
        if source_path_match:
            source_path = args.disk
        else:
            project = properties.VALUES.core.project.Get(required=True)
            source_path = 'projects/' + project + '/zones/' + args.zone + '/disks/' + args.disk
        source_disk_list = []
        for disk in node.dataDisks:
            source_disk_list.append(disk.sourceDisk)
        for i, source_disk in enumerate(source_disk_list):
            if source_path != source_disk:
                continue
            if source_path == source_disk:
                del node.dataDisks[i]
                break
        else:
            raise exceptions.BadArgumentException('TPU', 'error: specified data disk is not currently attached to the TPU VM.')
        return tpu.UpdateNode(tpu_name_ref.Name(), args.zone, node, 'data_disks', 'Detaching disk from TPU VM')