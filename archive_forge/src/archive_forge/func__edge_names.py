from typing import Dict, List, MutableMapping, Optional, Set, Tuple
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils
def _edge_names(graph: GraphProto, exclude: Optional[Set[str]]=None) -> List[str]:
    if exclude is None:
        exclude = set()
    edges = []
    for n in graph.node:
        for i in n.input:
            if i != '' and i not in exclude:
                edges.append(i)
        for o in n.output:
            if o != '' and o not in exclude:
                edges.append(o)
    return edges