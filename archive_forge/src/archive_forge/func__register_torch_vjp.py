from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
@lru_cache(maxsize=None)
def _register_torch_vjp():
    """
    Register the custom VJP for torch
    """
    import torch

    class _TorchFidelity(torch.autograd.Function):

        @staticmethod
        def forward(ctx, dm0, dm1):
            """Forward pass for _compute_fidelity"""
            fid = _compute_fidelity_vanilla(dm0, dm1)
            ctx.save_for_backward(dm0, dm1)
            return fid

        @staticmethod
        def backward(ctx, grad_out):
            """Backward pass for _compute_fidelity"""
            dm0, dm1 = ctx.saved_tensors
            return _compute_fidelity_grad(dm0, dm1, grad_out)
    ar.register_function('torch', 'compute_fidelity', _TorchFidelity.apply)