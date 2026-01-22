from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class GraphModule:
    graph: Graph
    signature: GraphSignature
    module_call_graph: List[ModuleCallEntry]