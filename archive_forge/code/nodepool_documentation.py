from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
Fill the node pool message from command arguments.

  Args:
    req: update node pool request message.
    messages: message module of edgecontainer node pool.
    args: command line arguments.
    update_mask_pieces: update mask pieces.
    existing_node_pool: existing node pool.
  