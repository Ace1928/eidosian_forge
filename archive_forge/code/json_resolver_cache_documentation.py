import warnings
import functools
from typing import Dict
from cirq.protocols.json_serialization import ObjectFactory
from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
Module for use in exporting cirq-google objects in JSON.