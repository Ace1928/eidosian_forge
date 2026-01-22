from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def UpdateLegacyLabels(unused_ref, args, patch_request):
    """Updates labels of connector."""
    labels_diff = labels_util.Diff.FromUpdateArgs(args)
    if labels_diff.MayHaveUpdates():
        patch_request = command_util.AddFieldToUpdateMask('labels', patch_request)
        messages = api_util.GetMessagesModule(args.calliope_command.ReleaseTrack())
        if patch_request.connector is None:
            patch_request.connector = messages.Connector()
        new_labels = labels_diff.Apply(messages.Connector.LabelsValue, patch_request.connector.labels).GetOrNone()
        if new_labels:
            patch_request.connector.labels = new_labels
    return patch_request