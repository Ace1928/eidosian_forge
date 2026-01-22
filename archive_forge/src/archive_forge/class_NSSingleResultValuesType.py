import enum
from typing import NamedTuple
from torch.fx.graph import Node
from typing import Dict, Any, List, Union, Callable
class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = 'weight'
    NODE_OUTPUT = 'node_output'
    NODE_INPUT = 'node_input'