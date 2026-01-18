from __future__ import annotations
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
Level 2 pass manager: medium optimization by initial layout selection and
    gate cancellation using commutativity rules.

    This pass manager applies the user-given initial layout. If none is given, a search
    for a perfect layout (i.e. one that satisfies all 2-qubit interactions) is conducted.
    If no such layout is found, qubits are laid out on the most densely connected subset
    which also exhibits the best gate fidelities.

    The pass manager then transforms the circuit to match the coupling constraints.
    It is then unrolled to the basis, and any flipped cx directions are fixed.
    Finally, optimizations in the form of commutative gate cancellation and redundant
    reset removal are performed.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 2 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    