import functools
import torch
import torch.distributed as dist
from typing import Optional
class DefaultState:
    """
    Stores state needed to perform the default communication algorithm within a communication hook.

    Args:
        process_group (ProcessGroup): The process group to be used.
    """
    __slots__ = ['process_group', 'world_size', 'gradient_predivide_factor', 'gradient_postdivide_factor']

    def __init__(self, process_group: dist.ProcessGroup):
        if process_group is None:
            raise ValueError(f'Expected to pass in an explicit ProcessGroup to {self}.')
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.gradient_predivide_factor = self._get_gradient_predivide_factor(self.world_size)
        self.gradient_postdivide_factor = self.world_size / self.gradient_predivide_factor

    @staticmethod
    def _get_gradient_predivide_factor(world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)