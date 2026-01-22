from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
import torch
from torch import Tensor, nn
from torch.distributed.rpc import RRef
import torch.autograd
import torch.cuda
from . import microbatch
from .batchnorm import DeferredBatchNorm
from .pipeline import Pipeline
from .skip.layout import inspect_skip_layout
from .skip.skippable import verify_skippables
from .stream import AbstractStream, new_stream
def _ensure_copy_streams(self) -> List[List[AbstractStream]]:
    """Ensures that :class:`Pipe` caches CUDA streams for copy.

        It's worth to cache CUDA streams although PyTorch already manages a
        pool of pre-allocated CUDA streams, because it may reduce GPU memory
        fragmentation when the number of micro-batches is small.

        """
    if not self._copy_streams:
        for device in self.devices:
            self._copy_streams.append([new_stream(device) for _ in range(self.chunks)])
    return self._copy_streams