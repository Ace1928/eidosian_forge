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
@dataclasses.dataclass(frozen=True)
class _DecomposeArgs:
    context: Optional[DecompositionContext]
    intercepting_decomposer: Optional[OpDecomposer]
    fallback_decomposer: Optional[OpDecomposer]
    keep: Optional[Callable[['cirq.Operation'], bool]]
    on_stuck_raise: Union[None, Exception, Callable[['cirq.Operation'], Optional[Exception]]]
    preserve_structure: bool