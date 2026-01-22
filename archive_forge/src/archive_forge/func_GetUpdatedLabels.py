from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.command_lib.util.args import labels_util
def GetUpdatedLabels(args, req, feature_name):
    """Return updated resource labels."""
    labels_diff = labels_util.Diff.FromUpdateArgs(args)
    if labels_diff.MayHaveUpdates():
        req = AddFieldToUpdateMask('labels', req)
        api_version = util.GetApiVersionFromArgs(args)
        messages = filestore_client.GetMessages(api_version)
        if feature_name == snapshot_feature_name:
            return labels_diff.Apply(messages.Snapshot.LabelsValue, req.snapshot.labels).GetOrNone()
        if feature_name == backup_feature_name:
            return labels_diff.Apply(messages.Backup.LabelsValue, req.backup.labels).GetOrNone()
    return None