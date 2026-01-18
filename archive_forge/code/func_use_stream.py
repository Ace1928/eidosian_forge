from contextlib import contextmanager
from typing import Generator, List, Union, cast
import torch
@contextmanager
def use_stream(stream: AbstractStream) -> Generator[None, None, None]:
    """:func:`torch.cuda.stream` for either CPU or CUDA stream."""
    if not is_cuda(stream):
        yield
        return
    with torch.cuda.stream(as_cuda(stream)):
        yield