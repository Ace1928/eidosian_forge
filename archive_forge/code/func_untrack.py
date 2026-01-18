from typing import List, Set, Tuple
from ray.autoscaler._private import constants
def untrack(self, node_id: str):
    """Gracefully stop tracking a node. If a node is intentionally removed from
        the cluster, we should stop tracking it so we don't mistakenly mark it
        as failed.

        Args:
            node_id: The node id which failed.
        """
    if node_id in self.node_mapping:
        self.lru_order.remove(node_id)
        del self.node_mapping[node_id]