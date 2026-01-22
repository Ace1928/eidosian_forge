import torch
from torch._export.db.case import export_case
from torch.export import Dim
from functorch.experimental.control_flow import cond

    The operands passed to cond() must be:
      - a list of tensors
      - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    