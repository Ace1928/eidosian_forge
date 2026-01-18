import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_param_sweep_size_no_sweeps():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[params_pb2.ZipSweep(), params_pb2.ZipSweep()]))
    assert len(params.sweep_from_proto(ps)) == 1