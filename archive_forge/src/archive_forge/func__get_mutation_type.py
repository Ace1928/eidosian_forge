import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def _get_mutation_type(keep_input_mutations: bool, mutates_data, mutates_metadata, mutations_hidden_from_autograd, mutations_under_no_grad_or_inference_mode, requires_grad):
    if not mutates_data and (not mutates_metadata):
        return MutationType.NOT_MUTATED
    if _check_if_mutation_can_be_in_graph(keep_input_mutations, mutates_data, mutates_metadata, mutations_hidden_from_autograd, mutations_under_no_grad_or_inference_mode, requires_grad):
        return MutationType.MUTATED_IN_GRAPH
    return MutationType.MUTATED_OUT_GRAPH