import torch
from torch._export.db.case import export_case

    A for loop with constant number of iterations should be unrolled in the exported graph.
    