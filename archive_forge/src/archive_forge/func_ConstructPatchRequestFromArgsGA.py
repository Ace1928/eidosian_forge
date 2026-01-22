from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def ConstructPatchRequestFromArgsGA(alloydb_messages, cluster_ref, args):
    """Returns the cluster patch request for GA release track based on args."""
    cluster, update_masks = _ConstructClusterAndMaskForPatchRequestGA(alloydb_messages, args)
    return alloydb_messages.AlloydbProjectsLocationsClustersPatchRequest(name=cluster_ref.RelativeName(), cluster=cluster, updateMask=','.join(update_masks))