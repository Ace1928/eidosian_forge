import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
def _foreach_non_finite_check_and_unscale_cpu_(self, grads: Sequence[torch.Tensor], found_inf: torch.Tensor, inv_scale: torch.Tensor) -> None:
    if len(grads) == 0:
        return
    assert inv_scale.numel() == 1, 'inv_scale must be a 1-element tensor.'
    assert found_inf.numel() == 1, 'found_inf must be a 1-element tensor.'
    for grad in grads:
        if grad.device.type != 'cpu':
            log.error('tensor device is %s but was expected to be ``cpu``', grad.device)
            raise ValueError('Gradients were found on a non-CPU device when expected to be on CPU.')
        if torch.isinf(grad).any().item() is True or torch.isnan(grad).any().item() is True:
            found_inf.data = torch.tensor([1.0])
            break
        else:
            grad.data *= inv_scale.item()