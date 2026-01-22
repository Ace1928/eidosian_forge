import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import map

    functorch map() maps a function over the first tensor dimension.
    