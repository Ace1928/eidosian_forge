import logging
import torch
import torch.distributed as dist
from . import default_hooks as default
class PostLocalSGDState:
    """
    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,
    and all-reducing gradients locally using ``subgroup`` afterwards.

    If ``process_group`` is ``None``, the global process group will be used.
    If ``subgroup`` is ``None``, the intra-node process group on each machine will be used.

    Additionally, ``post_local_gradient_allreduce`` may be worth tuning,
    because both true and false may give a faster convergence.
    """
    __slots__ = ['process_group', 'subgroup', 'start_localSGD_iter', 'post_local_gradient_allreduce', 'iter']

    def __init__(self, process_group, subgroup, start_localSGD_iter, post_local_gradient_allreduce=True):
        logger.info('Local SGD will be started after %s iterations', start_localSGD_iter)
        self.process_group = process_group
        self.subgroup = subgroup
        self.start_localSGD_iter = start_localSGD_iter
        self.post_local_gradient_allreduce = post_local_gradient_allreduce
        self.iter = 0

    def maybe_increase_iter(self, bucket):
        if bucket.is_last():
            self.iter += 1
        if self.iter == self.start_localSGD_iter:
            logger.info('Start to apply local SGD after %s iterations.', self.iter)