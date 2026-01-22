from typing import Any, TypeVar, Optional, Sequence, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import unitary_protocol
@doc_private
def _trace_distance_bound_(self) -> float:
    """A maximum on the trace distance between `val`'s input and output.

        Generally this method is used when deciding whether to keep gates, so
        only the behavior near 0 is important. Approximations that overestimate
        the maximum trace distance are permitted. If, for any case, the bound
        exceeds 1, this function will return 1.  Underestimates are not
        permitted.
        """