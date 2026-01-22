from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.interconnects import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.interconnects import flags
from googlecloudsdk.command_lib.util.args import labels_util
def _DoRun(self, args, support_labels=False):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    ref = self.INTERCONNECT_ARG.ResolveAsResource(args, holder.resources)
    interconnect = client.Interconnect(ref, compute_client=holder.client)
    labels = None
    label_fingerprint = None
    if support_labels:
        labels_diff = labels_util.Diff.FromUpdateArgs(args)
        if labels_diff.MayHaveUpdates():
            old_interconnect = interconnect.Describe()
            labels = labels_diff.Apply(holder.client.messages.Interconnect.LabelsValue, old_interconnect.labels).GetOrNone()
            if labels is not None:
                label_fingerprint = old_interconnect.labelFingerprint
    return interconnect.Patch(description=args.description, interconnect_type=None, requested_link_count=args.requested_link_count, link_type=None, admin_enabled=args.admin_enabled, noc_contact_email=args.noc_contact_email, location=None, labels=labels, label_fingerprint=label_fingerprint)