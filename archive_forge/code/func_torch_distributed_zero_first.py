from ..utils import DummyObject, requires_backends
def torch_distributed_zero_first(*args, **kwargs):
    requires_backends(torch_distributed_zero_first, ['torch'])