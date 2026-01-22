from typing import AbstractSet, Any, Optional, Tuple
import numpy as np
import cirq
from cirq._compat import proper_repr
def _resolve_parameters_(self, resolver: cirq.ParamResolverOrSimilarType, recursive: bool) -> 'CouplerPulse':
    return CouplerPulse(hold_time=cirq.resolve_parameters(self.hold_time, resolver, recursive=recursive), coupling_mhz=cirq.resolve_parameters(self.coupling_mhz, resolver, recursive=recursive), rise_time=cirq.resolve_parameters(self.rise_time, resolver, recursive=recursive), padding_time=cirq.resolve_parameters(self.padding_time, resolver, recursive=recursive), q0_detune_mhz=cirq.resolve_parameters(self.q0_detune_mhz, resolver, recursive=recursive), q1_detune_mhz=cirq.resolve_parameters(self.q1_detune_mhz, resolver, recursive=recursive))