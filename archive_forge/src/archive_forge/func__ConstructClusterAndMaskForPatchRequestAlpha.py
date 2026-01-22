from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructClusterAndMaskForPatchRequestAlpha(alloydb_messages, args):
    """Returns the cluster resource for patch request."""
    cluster, update_masks = _ConstructClusterAndMaskForPatchRequestBeta(alloydb_messages, args)
    return (cluster, update_masks)