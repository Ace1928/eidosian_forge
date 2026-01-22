from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import labels_doc_helper
from googlecloudsdk.command_lib.compute import labels_flags
from googlecloudsdk.command_lib.compute.images import flags as images_flags
from googlecloudsdk.command_lib.util.args import labels_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ImagesAddLabels(base.UpdateCommand):
    DISK_IMAGE_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.DISK_IMAGE_ARG = images_flags.MakeDiskImageArg(plural=False)
        cls.DISK_IMAGE_ARG.AddArgument(parser)
        labels_flags.AddArgsForAddLabels(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client.apitools_client
        messages = holder.client.messages
        image_ref = self.DISK_IMAGE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetDefaultScopeLister(holder.client))
        add_labels = labels_util.GetUpdateLabelsDictFromArgs(args)
        image = client.images.Get(messages.ComputeImagesGetRequest(**image_ref.AsDict()))
        labels_update = labels_util.Diff(additions=add_labels).Apply(messages.GlobalSetLabelsRequest.LabelsValue, image.labels)
        if not labels_update.needs_update:
            return image
        request = messages.ComputeImagesSetLabelsRequest(project=image_ref.project, resource=image_ref.image, globalSetLabelsRequest=messages.GlobalSetLabelsRequest(labelFingerprint=image.labelFingerprint, labels=labels_update.labels))
        operation = client.images.SetLabels(request)
        operation_ref = holder.resources.Parse(operation.selfLink, collection='compute.globalOperations')
        operation_poller = poller.Poller(client.images)
        return waiter.WaitFor(operation_poller, operation_ref, 'Updating labels of image [{0}]'.format(image_ref.Name()))