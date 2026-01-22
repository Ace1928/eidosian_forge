import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond

    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - torch.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    