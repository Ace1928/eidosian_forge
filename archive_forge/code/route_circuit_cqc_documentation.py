from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
Computes the cost function for the given list of swaps over the current timestep ops.

        To use this transformer with a different cost function, create a new subclass that derives
        from `RouteCQC` and override this method.
        