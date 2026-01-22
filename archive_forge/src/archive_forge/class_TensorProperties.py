from dataclasses import dataclass, field
from enum import Enum
from typing import List
import torch
from torch.distributed._shard.metadata import ShardMetadata
@dataclass
class TensorProperties:
    """ Properties used to create :class:`Tensor` """
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self):
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f'Invalid torch.memory_format: {memory_format}')
        return (self.dtype, self.layout, self.requires_grad, mem_format_encoding, self.pin_memory)

    def __setstate__(self, state):
        self.dtype, self.layout, self.requires_grad, mem_format_encoding, self.pin_memory = state
        if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(f'Invalid torch.memory_format encoding: {mem_format_encoding}')
        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> 'TensorProperties':
        return TensorProperties(dtype=tensor.dtype, layout=tensor.layout, requires_grad=tensor.requires_grad, memory_format=torch.contiguous_format, pin_memory=tensor.is_pinned())