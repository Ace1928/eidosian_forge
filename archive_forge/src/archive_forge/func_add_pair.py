import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def add_pair(neighbor: 'cirq.GridQubit'):
    if neighbor in qubits:
        pairs.append((qubit, neighbor))