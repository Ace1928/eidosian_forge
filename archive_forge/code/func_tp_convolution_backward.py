from typing import cast, Dict, List, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
def tp_convolution_backward(op_call: torch._ops.OpOverload, local_tensor_args: Tuple[object, ...], local_tensor_kwargs: Dict[str, object]) -> object:
    assert op_call == aten.convolution_backward.default
    assert len(local_tensor_args) == 11
    rank = dist.get_rank()
    size = dist.get_world_size()
    grad_out_tensor = cast(torch.Tensor, local_tensor_args[0])
    in_tensor = cast(torch.Tensor, local_tensor_args[1])
    weight = cast(torch.Tensor, local_tensor_args[2])
    stride, padding, dilation = local_tensor_args[4:7]
    assert _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation)
    assert isinstance(padding, List)
    if not _requires_data_exchange(padding):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        d = weight.shape[3] - 1
        d1 = d // 2
        d2 = d - d1
        assert d1 + d2 == d
        right = (rank + 1) % size
        left = (rank - 1 + size) % size
        in_tensor = _ring_send_recv_construct(in_tensor, d1, d2, left, right, rank, size)
        N, C_out, H_out, _ = grad_out_tensor.shape
        padding_w = padding[1]
        if rank == 0:
            grad_out_tensor = torch.nn.functional.pad(grad_out_tensor, (0, padding_w), 'constant', 0)
        elif rank == size - 1:
            grad_out_tensor = torch.nn.functional.pad(grad_out_tensor, (padding_w, 0), 'constant', 0)
        else:
            grad_out_tensor = torch.nn.functional.pad(grad_out_tensor, (padding_w, padding_w), 'constant', 0)
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = grad_out_tensor
        local_tensor_args_list[1] = in_tensor
        local_tensor_args = cast(Tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        grad_in_tensor = local_results[0]
        grad_in_tensor = _ring_send_recv_aggregate(grad_in_tensor, d1, d2, left, right, rank, size)
        local_results = list(local_results)
        local_results[0] = grad_in_tensor
        local_results = cast(Tuple[object, ...], local_results)
        return local_results