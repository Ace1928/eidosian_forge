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
@runtime_checkable
class OpDecomposerWithContext(Protocol):

    def __call__(self, __op: 'cirq.Operation', *, context: Optional['cirq.DecompositionContext']=None) -> DecomposeResult:
        ...