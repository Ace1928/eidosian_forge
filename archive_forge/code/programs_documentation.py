import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
Convert the proto to the corresponding operation.

    See protos in api/google/v1 for specification of the protos.

    Args:
        proto: Operation proto.

    Returns:
        The operation.

    Raises:
        ValueError: If the proto has an operation that is invalid.
    