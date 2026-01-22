from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@dataclass
class CompileToNativeQuilRequest:
    """
    Request to compile to native Quil.
    """
    program: str
    'Program to compile.'
    target_quantum_processor: TargetQuantumProcessor
    'Quantum processor to target.'
    protoquil: Optional[bool]
    'Whether or not to restrict to protoquil. Overrides server default when provided.'