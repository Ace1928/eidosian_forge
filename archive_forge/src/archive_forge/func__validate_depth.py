from typing import Callable, Sequence, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2
def _validate_depth(circuits: Sequence[cirq.AbstractCircuit], max_moments: int=MAX_MOMENTS) -> None:
    """Validate that the depth of the circuit is not too long (too many moments)."""
    for circuit in circuits:
        if len(circuit) > max_moments:
            raise RuntimeError(f'Provided circuit exceeds the limit of {max_moments} moments.')