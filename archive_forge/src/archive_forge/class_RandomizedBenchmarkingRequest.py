import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class RandomizedBenchmarkingRequest(Message):
    """
    RPC request payload for generating a randomized benchmarking sequence.
    """
    depth: int
    'Depth of the benchmarking sequence.'
    qubits: int
    'Number of qubits involved in the benchmarking sequence.'
    gateset: List[str]
    'List of Quil programs, each describing a Clifford.'
    seed: Optional[int] = None
    'PRNG seed. Set this to guarantee repeatable results.'
    interleaver: Optional[str] = None
    'Fixed Clifford, specified as a Quil string, to interleave through an RB sequence.'