from typing import List, Set, Tuple
from ray.autoscaler._private import constants
def _add_node_mapping(self, node_id: str, value: str):
    if node_id in self.node_mapping:
        return
    assert len(self.lru_order) == len(self.node_mapping)
    if len(self.lru_order) >= constants.AUTOSCALER_MAX_NODES_TRACKED:
        node_id = self.lru_order.pop(0)
        del self.node_mapping[node_id]
    self.node_mapping[node_id] = value
    self.lru_order.append(node_id)