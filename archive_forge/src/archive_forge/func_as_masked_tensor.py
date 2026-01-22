from .core import MaskedTensor
def as_masked_tensor(data, mask):
    return MaskedTensor._from_values(data, mask)