from typing import Any, Optional
from langchain_core.runnables.graph import Graph, LabelsDict
def update_styles(self, viz: Any, graph: Graph) -> None:
    if (first := graph.first_node()):
        viz.get_node(first.id).attr.update(fillcolor='lightblue')
    if (last := graph.last_node()):
        viz.get_node(last.id).attr.update(fillcolor='orange')