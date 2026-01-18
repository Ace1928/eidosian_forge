import torch
from torch._export.db.case import export_case, SupportLevel, export_rewrite_case
@export_rewrite_case(parent=type_reflection_method)
def type_reflection_method_rewrite(x):
    """
    Custom object class methods will be inlined.
    """
    return A.func(x)