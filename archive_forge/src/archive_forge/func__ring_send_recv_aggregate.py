from typing import cast, Dict, List, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
def _ring_send_recv_aggregate(grad_in_tensor, d1, d2, left, right, rank, size):
    send_to_right = grad_in_tensor[:, :, :, -d2:].contiguous()
    send_to_left = grad_in_tensor[:, :, :, :d1].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)
    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)
    reqs = dist.batch_isend_irecv([send_op_right, send_op_left, recv_op_left, recv_op_right])
    for req in reqs:
        req.wait()
    if rank == 0:
        grad_in_tensor = grad_in_tensor[:, :, :, :-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(grad_in_tensor[:, :, :, -d1:], recv_from_right)
    elif rank == size - 1:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:]
        grad_in_tensor[:, :, :, :d2] = torch.add(grad_in_tensor[:, :, :, :d2], recv_from_left)
    else:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(grad_in_tensor[:, :, :, -d1:], recv_from_right)
        grad_in_tensor[:, :, :, :d2] = torch.add(grad_in_tensor[:, :, :, :d2], recv_from_left)