import torch
class _PositiveSemidefinite(_Symmetric):
    """
    Constrain to positive-semidefinite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.eigvalsh(value).ge(0).all(-1)