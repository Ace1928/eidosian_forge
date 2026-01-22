import torch
import torch.distributed as dist
from torch.autograd.function import Function
class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        if not (input.is_contiguous(memory_format=torch.channels_last) or input.is_contiguous(memory_format=torch.channels_last_3d)):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        size = int(input.numel() // input.size(1))
        if size == 1 and world_size < 2:
            raise ValueError(f'Expected more than 1 value per channel when training, got input size {size}')
        num_channels = input.shape[1]
        if input.numel() > 0:
            mean, invstd = torch.batch_norm_stats(input, eps)
            count = torch.full((1,), input.numel() // input.size(1), dtype=mean.dtype, device=mean.device)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            combined = torch.zeros(2 * num_channels + 1, dtype=input.dtype, device=input.device)
        if process_group._get_backend_name() != 'gloo':
            combined_size = combined.numel()
            combined_flat = torch.empty(1, combined_size * world_size, dtype=combined.dtype, device=combined.device)
            dist.all_gather_into_tensor(combined_flat, combined, process_group, async_op=False)
            combined = torch.reshape(combined_flat, (world_size, combined_size))
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        else:
            combined_list = [torch.empty_like(combined) for _ in range(world_size)]
            dist.all_gather(combined_list, combined, process_group, async_op=False)
            combined = torch.stack(combined_list, dim=0)
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            mask = count_all.squeeze(-1) >= 1
            count_all = count_all[mask]
            mean_all = mean_all[mask]
            invstd_all = invstd_all[mask]
        counts = count_all.view(-1)
        if running_mean is not None and counts.dtype != running_mean.dtype:
            counts = counts.to(running_mean.dtype)
        mean, invstd = torch.batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, counts)
        self.save_for_backward(input, weight, mean, invstd, count_all.to(torch.int32))
        self.process_group = process_group
        if input.numel() > 0:
            return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        else:
            return torch.empty_like(input)

    @staticmethod
    def backward(self, grad_output):
        if not (grad_output.is_contiguous(memory_format=torch.channels_last) or grad_output.is_contiguous(memory_format=torch.channels_last_3d)):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        if saved_input.numel() > 0:
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd, weight, self.needs_input_grad[0], self.needs_input_grad[1], self.needs_input_grad[2])
            if self.needs_input_grad[0]:
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)
                if weight is not None and weight.dtype != mean.dtype:
                    weight = weight.to(mean.dtype)
                grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor)
            if weight is None or not self.needs_input_grad[1]:
                grad_weight = None
            if weight is None or not self.needs_input_grad[2]:
                grad_bias = None
        else:
            num_channels = saved_input.shape[1]
            if self.needs_input_grad[0]:
                combined = torch.zeros(2 * num_channels, dtype=saved_input.dtype, device=saved_input.device)
                torch.distributed.all_reduce(combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
        return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None)