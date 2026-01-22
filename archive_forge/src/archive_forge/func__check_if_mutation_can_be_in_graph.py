import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def _check_if_mutation_can_be_in_graph(keep_input_mutations: bool, mutates_data, mutates_metadata, mutations_hidden_from_autograd, mutations_under_no_grad_or_inference_mode, requires_grad):
    if keep_input_mutations:
        return mutates_data and (not mutates_metadata and (not requires_grad) or mutations_hidden_from_autograd or mutations_under_no_grad_or_inference_mode)
    return False