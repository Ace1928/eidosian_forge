from contextlib import contextmanager
from typing import Generator, List, Union, cast
import torch
def wait_stream(source: AbstractStream, target: AbstractStream) -> None:
    """:meth:`torch.cuda.Stream.wait_stream` for either CPU or CUDA stream. It
    makes the source stream wait until the target stream completes work queued.
    """
    if is_cuda(target):
        if is_cuda(source):
            as_cuda(source).wait_stream(as_cuda(target))
        else:
            as_cuda(target).synchronize()