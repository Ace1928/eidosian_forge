import base64
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from langchain_core.runnables.graph import (
def _adjust_mermaid_edge(edge: Edge, nodes: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Adjusts Mermaid edge to map conditional nodes to pure nodes."""
    source_node_label = nodes.get(edge.source, edge.source)
    target_node_label = nodes.get(edge.target, edge.target)
    return (source_node_label, target_node_label)