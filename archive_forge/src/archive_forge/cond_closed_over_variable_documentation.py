import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond

    torch.cond() supports branches closed over arbitrary variables.
    