import itertools
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch

        Run the complete algorithm for finding all maximal matches for the given template and
        circuit. First it fixes the configuration of the circuit due to the first match.
        Then it explores all compatible qubit configurations of the circuit. For each
        qubit configurations, we apply first the Forward part of the algorithm  and then
        the Backward part of the algorithm. The longest matches for the given configuration
        are stored. Finally, the list of stored matches is sorted.
        