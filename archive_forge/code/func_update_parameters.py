import itertools
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def update_parameters(self, model):
    self_param = itertools.chain(self.module.parameters(), self.module.buffers()) if self.use_buffers else self.parameters()
    model_param = itertools.chain(model.parameters(), model.buffers()) if self.use_buffers else model.parameters()
    self_param_detached = []
    model_param_detached = []
    for p_averaged, p_model in zip(self_param, model_param):
        p_model_ = p_model.detach().to(p_averaged.device)
        self_param_detached.append(p_averaged.detach())
        model_param_detached.append(p_model_)
        if self.n_averaged == 0:
            p_averaged.detach().copy_(p_model_)
    if self.n_averaged > 0:
        if self.multi_avg_fn is not None or self.avg_fn is None:
            grouped_tensors = _group_tensors_by_device_and_dtype([self_param_detached, model_param_detached])
            for (device, _), ([self_params, model_params], _) in grouped_tensors.items():
                if self.multi_avg_fn:
                    self.multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                elif device.type in _get_foreach_kernels_supported_devices():
                    multi_avg_fn = get_swa_multi_avg_fn()
                    multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                else:
                    avg_fn = get_swa_avg_fn()
                    n_averaged = self.n_averaged.to(device)
                    for p_averaged, p_model in zip(self_params, model_params):
                        p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
        else:
            for p_averaged, p_model in zip(self_param_detached, model_param_detached):
                n_averaged = self.n_averaged.to(p_averaged.device)
                p_averaged.detach().copy_(self.avg_fn(p_averaged.detach(), p_model, n_averaged))
    if not self.use_buffers:
        for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
            b_swa.detach().copy_(b_model.detach().to(b_swa.device))
    self.n_averaged += 1