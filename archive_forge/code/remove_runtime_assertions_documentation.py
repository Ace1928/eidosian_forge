import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

    Remove runtime assertions inserted by the
    _AddRuntimeAssertionsForInlineConstraintsPass.
    