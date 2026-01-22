from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@dataclass
class CompileToNativeQuilResponse:
    """
    Compile to native Quil response.
    """
    native_program: str
    'Native Quil program.'
    metadata: Optional[NativeQuilMetadataResponse]
    'Metadata for the returned Native Quil.'