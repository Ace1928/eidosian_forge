import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def _handle_kernel_launch(self, stream: StreamId, read_only: Set[DataPtr], read_write: Set[DataPtr], outputs: Set[DataPtr], operator: str, tensor_aliases: Dict[int, List[str]]) -> List[SynchronizationError]:

    def check_conflict(data_ptr: DataPtr, current_access: Access, previous_access: Optional[Access]) -> None:
        if previous_access is None:
            return
        if not self.syncs.is_ordered_after(current_access.stream, previous_access.seq_num, previous_access.stream):
            error_list.append(UnsynchronizedAccessError(data_ptr, self.tensors_accessed.get_allocation_stack_trace(data_ptr), current_access, previous_access))
    error_list: List[SynchronizationError] = []
    self.seq_num += 1
    self.syncs.update_seq_num(stream, self.seq_num)
    stack_trace = traceback.StackSummary.extract(traceback.walk_stack(inspect.currentframe()), lookup_lines=False)
    stack_trace.reverse()
    for data_ptr in read_only:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        current_access = Access(AccessType.READ, self.seq_num, stream, operator, tensor_aliases[data_ptr], data_ptr in outputs, stack_trace)
        check_conflict(data_ptr, current_access, self.tensors_accessed.get_write(data_ptr))
        self.tensors_accessed.add_read(data_ptr, current_access)
    for data_ptr in read_write:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        current_access = Access(AccessType.WRITE, self.seq_num, stream, operator, tensor_aliases[data_ptr], data_ptr in outputs, stack_trace)
        if self.tensors_accessed.were_there_reads_since_last_write(data_ptr):
            for previous_access in self.tensors_accessed.get_reads(data_ptr):
                check_conflict(data_ptr, current_access, previous_access)
        else:
            check_conflict(data_ptr, current_access, self.tensors_accessed.get_write(data_ptr))
        self.tensors_accessed.set_write(data_ptr, current_access)
    return error_list