from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def _get_device_graph(device_or_qubits: Any):
    qubits = device_or_qubits if isinstance(device_or_qubits, list) else device_or_qubits.qubits
    return ccr.gridqubits_to_graph_device(qubits)