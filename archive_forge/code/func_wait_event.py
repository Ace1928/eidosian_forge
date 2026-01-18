import ctypes
import torch
from torch._streambase import _EventBase, _StreamBase
from ._utils import _dummy_type
def wait_event(self, event):
    """Make all future work submitted to the stream wait for an event.

        Args:
            event (torch.cuda.Event): an event to wait for.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
           `CUDA Stream documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _CUDA Stream documentation:
           https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
    event.wait(self)