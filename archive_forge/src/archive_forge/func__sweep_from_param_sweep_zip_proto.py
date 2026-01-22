from typing import cast
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.api.v1 import params_pb2
def _sweep_from_param_sweep_zip_proto(param_sweep_zip: params_pb2.ZipSweep) -> cirq.Sweep:
    if len(param_sweep_zip.sweeps) > 0:
        return cirq.Zip(*[_sweep_from_single_param_sweep_proto(sweep) for sweep in param_sweep_zip.sweeps])
    return cirq.UnitSweep