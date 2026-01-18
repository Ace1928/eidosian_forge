from typing import Dict, List, Tuple
from torch.fx import Graph, GraphModule, Node
from torch.fx._compatibility import compatibility
from .matcher_utils import InternalMatch, SubgraphMatcher
The returned InternalMatch will have name_node_map populated with a map
        from node name (str) to the target node, e.g.
        {"conv": target_conv_ndoe, "relu": target_relu_node}

        this requires the pattern graph returns an additional
        output of node name to node, e.g. instead of:
        ```
        def pattern(...):
            ...
            return relu
        ```
        we should do:
        ```
        def pattern(...):
            ...
            return relu, {"conv": conv, "relu": relu}
        ``` instead
        