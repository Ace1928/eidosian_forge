from __future__ import annotations
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.circuit.gate import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevDecomposition
from qiskit.synthesis.discrete_basis.generate_basis_approximations import (
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from .plugin import UnitarySynthesisPlugin
class SolovayKitaevSynthesis(UnitarySynthesisPlugin):
    """A Solovay-Kitaev Qiskit unitary synthesis plugin.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"sk"``.

    This plugin supports customization and additional parameters can be passed to the plugin
    by passing a dictionary as the ``unitary_synthesis_plugin_config`` parameter of
    the :func:`~qiskit.compiler.transpile` function.

    Supported parameters in the dictionary:

    basis_approximations (str | dict):
        The basic approximations for the finding the best discrete decomposition at the root of the
        recursion. If a string, it specifies the ``.npy`` file to load the approximations from.
        If a dictionary, it contains ``{label: SO(3)-matrix}`` pairs. If None, a default based on
        the specified ``basis_gates`` and ``depth`` is generated.

    basis_gates (list):
        A list of strings specifying the discrete basis gates to decompose to. If None,
        defaults to ``["h", "t", "tdg"]``.

    depth (int):
        The gate-depth of the basic approximations. All possible, unique combinations of the
        basis gates up to length ``depth`` are considered. If None, defaults to 10.

    recursion_degree (int):
        The number of times the decomposition is recursively improved. If None, defaults to 3.
    """
    _sk = None

    @property
    def max_qubits(self):
        """Maximum number of supported qubits is ``1``."""
        return 1

    @property
    def min_qubits(self):
        """Minimum number of supported qubits is ``1``."""
        return 1

    @property
    def supports_natural_direction(self):
        """The plugin does not support natural direction, it does not assume
        bidirectional two qubit gates."""
        return True

    @property
    def supports_pulse_optimize(self):
        """The plugin does not support optimization of pulses."""
        return False

    @property
    def supports_gate_lengths(self):
        """The plugin does not support gate lengths."""
        return False

    @property
    def supports_gate_errors(self):
        """The plugin does not support gate errors."""
        return False

    @property
    def supported_bases(self):
        """The plugin does not support bases for synthesis."""
        return None

    @property
    def supports_basis_gates(self):
        """The plugin does not support basis gates. By default it synthesis to the
        ``["h", "t", "tdg"]`` gate basis."""
        return True

    @property
    def supports_coupling_map(self):
        """The plugin does not support coupling maps."""
        return False

    def run(self, unitary, **options):
        config = options.get('config') or {}
        recursion_degree = config.get('recursion_degree', 3)
        if SolovayKitaevSynthesis._sk is None:
            basic_approximations = config.get('basic_approximations', None)
            basis_gates = options.get('basis_gates', ['h', 't', 'tdg'])
            if basic_approximations is None:
                depth = config.get('depth', 10)
                basic_approximations = generate_basic_approximations(basis_gates, depth)
            SolovayKitaevSynthesis._sk = SolovayKitaevDecomposition(basic_approximations)
        approximate_circuit = SolovayKitaevSynthesis._sk.run(unitary, recursion_degree)
        dag_circuit = circuit_to_dag(approximate_circuit)
        return dag_circuit