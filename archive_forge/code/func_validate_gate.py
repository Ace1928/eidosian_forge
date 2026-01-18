import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
def validate_gate(self, gate: cirq.Gate):
    if gate not in self.metadata.gateset:
        raise ValueError(f'Unsupported gate type: {gate!r}')