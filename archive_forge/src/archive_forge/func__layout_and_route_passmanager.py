import copy
import dataclasses
import logging
import functools
import time
import numpy as np
import rustworkx as rx
from qiskit.converters import dag_to_circuit
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.sabre_layout import sabre_layout_and_routing
from qiskit._accelerate.sabre_swap import (
from qiskit.transpiler.passes.routing.sabre_swap import _build_sabre_dag, _apply_sabre_result
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.utils.parallel import CPU_COUNT
def _layout_and_route_passmanager(self, initial_layout):
    """Return a passmanager for a full layout and routing.

        We use a factory to remove potential statefulness of passes.
        """
    layout_and_route = [SetLayout(initial_layout), FullAncillaAllocation(self.coupling_map), EnlargeWithAncilla(), ApplyLayout(), self.routing_pass]
    pm = PassManager(layout_and_route)
    return pm