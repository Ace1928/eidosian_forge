from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
def GetNodePoolCreateRequest(args, release_track):
    """Get node pool create request message.

  Args:
    args: comand line arguments.
    release_track: release track of the command.

  Returns:
    message obj, node pool create request message.
  """
    messages = util.GetMessagesModule(release_track)
    node_pool_ref = GetNodePoolReference(args)
    req = messages.EdgecontainerProjectsLocationsClustersNodePoolsCreateRequest(nodePool=messages.NodePool(), nodePoolId=node_pool_ref.nodePoolsId, parent=node_pool_ref.Parent().RelativeName())
    PopulateNodePoolCreateMessage(req, messages, args)
    return req