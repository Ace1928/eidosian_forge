import torch
class _Square(Constraint):
    """
    Constrain to square matrices.
    """
    event_dim = 2

    def check(self, value):
        return torch.full(size=value.shape[:-2], fill_value=value.shape[-2] == value.shape[-1], dtype=torch.bool, device=value.device)