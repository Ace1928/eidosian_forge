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
class CUDASanitizerDispatchMode(TorchDispatchMode):

    def __init__(self):
        self.event_handler = EventHandler()
        torch._C._activate_cuda_trace()
        cuda_trace.register_callback_for_cuda_event_creation(self.event_handler._handle_event_creation)
        cuda_trace.register_callback_for_cuda_event_deletion(self.event_handler._handle_event_deletion)
        cuda_trace.register_callback_for_cuda_event_record(self.event_handler._handle_event_record)
        cuda_trace.register_callback_for_cuda_event_wait(self.event_handler._handle_event_wait)
        cuda_trace.register_callback_for_cuda_memory_allocation(self.event_handler._handle_memory_allocation)
        cuda_trace.register_callback_for_cuda_memory_deallocation(self.event_handler._handle_memory_deallocation)
        cuda_trace.register_callback_for_cuda_stream_creation(self.event_handler._handle_stream_creation)
        cuda_trace.register_callback_for_cuda_device_synchronization(self.event_handler._handle_device_synchronization)
        cuda_trace.register_callback_for_cuda_stream_synchronization(self.event_handler._handle_stream_synchronization)
        cuda_trace.register_callback_for_cuda_event_synchronization(self.event_handler._handle_event_synchronization)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        argument_handler = ArgumentHandler()
        argument_handler.parse_inputs(func._schema, args, kwargs)
        outputs = func(*args, **kwargs)
        argument_handler.parse_outputs(outputs)
        errors = self.event_handler._handle_kernel_launch(torch.cuda.current_stream().cuda_stream, argument_handler.dataptrs_read - argument_handler.dataptrs_written, argument_handler.dataptrs_written, argument_handler.outputs, func._schema, argument_handler.tensor_aliases)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise CUDASanitizerErrors(errors)
        return outputs