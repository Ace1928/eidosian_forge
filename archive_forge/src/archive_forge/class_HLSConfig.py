from typing import Optional, Union, List, Tuple
import rustworkx as rx
from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, ControlledGate, EquivalenceLibrary
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.circuit.annotated_operation import (
from qiskit.synthesis.clifford import (
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.permutation import (
from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin
class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level objects.

    A higher-level object is an object of type :class:`~.Operation` (e.g., :class:`.Clifford` or
    :class:`.LinearFunction`).  Each object is referred to by its :attr:`~.Operation.name` field
    (e.g., ``"clifford"`` for :class:`.Clifford` objects), and the applicable synthesis methods are
    tied to this name.

    In the config, each method is specified in one of several ways:

    1. a tuple consisting of the name of a known synthesis plugin and a dictionary providing
       additional arguments for the algorithm.
    2. a tuple consisting of an instance of :class:`.HighLevelSynthesisPlugin` and additional
       arguments for the algorithm.
    3. a single string of a known synthesis plugin
    4. a single instance of :class:`.HighLevelSynthesisPlugin`.

    The following example illustrates different ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis plugins should be declared in ``entry-points`` table for
    ``qiskit.synthesis`` in ``pyproject.toml``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the documentation for
    :class:`~.HighLevelSynthesis`.

    For an overview of the complete process of using high-level synthesis, see
    :ref:`using-high-level-synthesis-plugins`.
    """

    def __init__(self, use_default_on_unspecified=True, **kwargs):
        """Creates a high-level-synthesis config.

        Args:
            use_default_on_unspecified (bool): if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.methods = {}
        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods