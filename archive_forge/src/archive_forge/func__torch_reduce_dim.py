import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _torch_reduce_dim(fn):

    def reduce_dim(self, dim, keepdim=False, dtype=None):
        if self.is_sparse:
            msg = f'The sparse version of {fn} is not implemented in reductions.\nIf you would like this operator to be supported, please file an issue for a feature request at https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\nIn the case that the semantics for the operator are not trivial, it would be appreciated to also include a proposal for the semantics.'
            warnings.warn(msg)
            return NotImplemented
        if not is_masked_tensor(self):
            raise TypeError('Input to reduce_dim must be a MaskedTensor')
        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask()
        if fn == 'all':
            result_data = masked_fn(data, dim=dim, keepdim=keepdim, mask=mask)
        else:
            result_data = masked_fn(self, dim=dim, keepdim=keepdim, dtype=dtype, mask=self.get_mask())
        return as_masked_tensor(result_data, _multidim_any(mask, dim, keepdim))
    return reduce_dim