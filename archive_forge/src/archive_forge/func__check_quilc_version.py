from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Union
from pyquil._memory import Memory
from pyquil._version import pyquil_version
from pyquil.api._compiler_client import CompilerClient, CompileToNativeQuilRequest
from pyquil.external.rpcq import compiler_isa_to_target_quantum_processor
from pyquil.parser import parse_program
from pyquil.paulis import PauliTerm
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import ExpressionDesignator, MemoryReference
from pyquil.quilbase import Gate
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import NativeQuilMetadata, ParameterAref, ParameterSpec
def _check_quilc_version(version: str) -> None:
    """
    Verify that there is no mismatch between pyquil and quilc versions.

    :param version: quilc version.
    """
    major, minor, _ = map(int, version.split('.'))
    if major == 1 and minor < 8:
        raise QuilcVersionMismatch(f'Must use quilc >= 1.8.0 with pyquil >= 2.8.0, but you have quilc {version} and pyquil {pyquil_version}')