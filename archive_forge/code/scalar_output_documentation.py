import torch
from torch._export.db.case import export_case
from torch.export import Dim

    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    