from enum import Enum
import logging
import inspect
import itertools
import time
from rustworkx import PyDiGraph, vf2_mapping, PyGraph
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.passes.layout import vf2_utils
def _target_match(node_a, node_b):
    if isinstance(node_a, set):
        return node_a.issuperset(node_b.keys())
    else:
        return set(node_a).issubset(node_b)