import torch
from torch._export.db.case import export_case

    `if` statement with backed dynamic shape predicate will be specialized into
    one particular branch and generate a guard. However, export will fail if the
    the dimension is marked as dynamic shape from higher level API.
    