import itertools
import dataclasses
import inspect
from collections import defaultdict
from typing import (
from typing_extensions import runtime_checkable
from typing_extensions import Protocol
from cirq import devices, ops
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
class SupportsDecompose(Protocol):
    """An object that can be decomposed into simpler operations.

    All decomposition methods should ultimately terminate on basic 1-qubit and
    2-qubit gates included by default in Cirq. Cirq does not make any guarantees
    about what the final gate set is. Currently, decompositions within Cirq
    happen to converge towards the X, Y, Z, CZ, PhasedX, specified-matrix gates,
    and others. This set will vary from release to release. Because of this
    variability, it is important for consumers of decomposition to look for
    generic properties of gates, such as "two qubit gate with a unitary matrix",
    instead of specific gate types such as CZ gates (though a consumer is
    of course free to handle CZ gates in a special way, and consumers can
    give an `intercepting_decomposer` to `cirq.decompose` that attempts to
    target a specific gate set).

    For example, `cirq.TOFFOLI` has a `_decompose_` method that returns a pair
    of Hadamard gates surrounding a `cirq.CCZ`. Although `cirq.CCZ` is not a
    1-qubit or 2-qubit operation, it specifies its own `_decompose_` method
    that only returns 1-qubit or 2-qubit operations. This means that iteratively
    decomposing `cirq.TOFFOLI` terminates in 1-qubit and 2-qubit operations, and
    so almost all decomposition-aware code will be able to handle `cirq.TOFFOLI`
    instances.

    Callers are responsible for iteratively decomposing until they are given
    operations that they understand. The `cirq.decompose` method is a simple way
    to do this, because it has logic to recursively decompose until a given
    `keep` predicate is satisfied.

    Code implementing `_decompose_` MUST NOT create cycles, such as a gate A
    decomposes into a gate B which decomposes back into gate A. This will result
    in infinite loops when calling `cirq.decompose`.

    It is permitted (though not recommended) for the chain of decompositions
    resulting from an operation to hit a dead end before reaching 1-qubit or
    2-qubit operations. When this happens, `cirq.decompose` will raise
    a `TypeError` by default, but can be configured to ignore the issue or
    raise a caller-provided error.
    """

    @doc_private
    def _decompose_(self) -> DecomposeResult:
        pass

    def _decompose_with_context_(self, *, context: Optional[DecompositionContext]=None) -> DecomposeResult:
        pass