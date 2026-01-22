from typing import (
import re
import warnings
from dataclasses import dataclass
import cirq
from cirq_google import ops
from cirq_google import transformers
from cirq_google.api import v2
from cirq_google.devices import known_devices
from cirq_google.experimental import ops as experimental_ops
def _build_compilation_target_gatesets(gateset: cirq.Gateset) -> Sequence[cirq.CompilationTargetGateset]:
    """Detects compilation target gatesets based on what gates are inside the gateset."""
    target_gatesets: List[cirq.CompilationTargetGateset] = []
    if all((gate_family in gateset.gates for gate_family in _CZ_TARGET_GATES)):
        target_gatesets.append(transformers.GoogleCZTargetGateset(additional_gates=list(gateset.gates - set(_CZ_TARGET_GATES))))
    if all((gate_family in gateset.gates for gate_family in _SYC_TARGET_GATES)):
        target_gatesets.append(transformers.SycamoreTargetGateset())
    if all((gate_family in gateset.gates for gate_family in _SQRT_ISWAP_TARGET_GATES)):
        target_gatesets.append(cirq.SqrtIswapTargetGateset(additional_gates=list(gateset.gates - set(_SQRT_ISWAP_TARGET_GATES))))
    return tuple(target_gatesets)