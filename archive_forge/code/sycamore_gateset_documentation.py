import itertools
from typing import cast, Any, Dict, List, Optional, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq_google import ops
from cirq_google.transformers.analytical_decompositions import two_qubit_to_sycamore
Inits `cirq_google.SycamoreTargetGateset`.

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            tabulation: If set, a tabulation for the Sycamore gate is used for decomposing Matrix
                gates. If unset, an analytic calculation is used for Matrix gates. In both cases,
                known decompositions for gates take priority over analytical / tabulation methods.
                To get `cirq.TwoQubitGateTabulation`, call `cirq.two_qubit_gate_product_tabulation`
                with a base gate (in this case, `cirq_google.SYC`) and a maximum infidelity.
        