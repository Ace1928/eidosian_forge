import torch
from .core import _map_mt_args_kwargs, _wrap_result
def _unary_helper(fn, args, kwargs, inplace):
    if len(kwargs) != 0:
        raise ValueError('MaskedTensor unary ops require that len(kwargs) == 0. If you need support for this, please open an issue on Github.')
    for a in args[1:]:
        if torch.is_tensor(a):
            raise TypeError('MaskedTensor unary ops do not support additional Tensor arguments')
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x._masked_mask)
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x._masked_data)
    if args[0].layout == torch.sparse_coo:
        data_args[0] = data_args[0].coalesce()
        s = data_args[0].size()
        i = data_args[0].indices()
        data_args[0] = data_args[0].coalesce().values()
        v = fn(*data_args)
        result_data = torch.sparse_coo_tensor(i, v, size=s)
    elif args[0].layout == torch.sparse_csr:
        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_csr_tensor(crow, col, v)
    else:
        result_data = fn(*data_args)
    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        return _wrap_result(result_data, mask_args[0])