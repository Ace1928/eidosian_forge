import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_param_sweep_keys():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='foo', points=params_pb2.Points(points=range(5))), params_pb2.SingleSweep(parameter_key='bar', points=params_pb2.Points(points=range(7)))]), params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='baz', points=params_pb2.Points(points=range(11))), params_pb2.SingleSweep(parameter_key='qux', points=params_pb2.Points(points=range(13)))])]))
    out = params.sweep_from_proto(ps)
    assert out.keys == ['foo', 'bar', 'baz', 'qux']