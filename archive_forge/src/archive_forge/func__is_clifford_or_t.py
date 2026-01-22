from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
def _is_clifford_or_t(stc: Any, fail_quietly: bool) -> Optional[TComplexity]:
    """Attempts to infer the type of a gate/operation as one of clifford, T or Rotation."""
    if not isinstance(stc, (cirq.Gate, cirq.Operation)):
        return None
    if isinstance(stc, cirq.ClassicallyControlledOperation):
        stc = stc.without_classical_controls()
    if cirq.num_qubits(stc) <= 2 and cirq.has_stabilizer_effect(stc):
        return TComplexity(clifford=1)
    if stc in _T_GATESET:
        return TComplexity(t=1)
    if cirq.num_qubits(stc) == 1 and cirq.has_unitary(stc):
        return TComplexity(rotations=1)
    return None