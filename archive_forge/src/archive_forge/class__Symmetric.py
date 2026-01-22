import torch
class _Symmetric(_Square):
    """
    Constrain to Symmetric square matrices.
    """

    def check(self, value):
        square_check = super().check(value)
        if not square_check.all():
            return square_check
        return torch.isclose(value, value.mT, atol=1e-06).all(-2).all(-1)