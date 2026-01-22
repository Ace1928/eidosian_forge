from ray.dag import DAGNode
import os
import tempfile
from ray.dag.utils import _DAGNodeNameGenerator
from ray.util.annotations import DeveloperAPI
def _dfs(node):
    nodes.append(node)
    for child_node in node._get_all_child_nodes():
        edges.append((child_node, node))
    return node