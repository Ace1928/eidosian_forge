from contextlib import contextmanager
import threading
from typing import Dict, Generator, List, Optional, Tuple
from torch import Tensor
from ..checkpoint import is_checkpointing
from ..dependency import fork, join
from ..microbatch import Batch
from ..stream import AbstractStream
from .layout import SkipLayout
from .namespace import Namespace
from .portal import Portal
class SkipTrackerThroughPotals(SkipTracker):
    """Tracks saved skip tensors through portals. The skip tensors will be
    hidden in portals so that the autograd engine does not need to track them.

    This tracker is only used when the training or evaluating module is wrapped
    with :class:`torchpipe.Pipe`.

    """

    def __init__(self, skip_layout: SkipLayout) -> None:
        super().__init__()
        self.skip_layout = skip_layout
        self.portals: Dict[Tuple[Namespace, str], Portal] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        """Saves the stashed skip tensor in a portal. The portal is then
        connected to the given micro-batch with :class:`Join`.
        """
        if not self.skip_layout.requires_copy(ns, name):
            super().save(batch, ns, name, tensor)
            return
        if (ns, name) not in self.portals:
            if is_checkpointing():
                tensor_life = 3
            else:
                tensor_life = 2
            portal = Portal(tensor, tensor_life)
            self.portals[ns, name] = portal
        else:
            portal = self.portals[ns, name]
            tensor_life = 1
            portal.put_tensor(tensor, tensor_life)
        phony = portal.blue()
        tensor_idx = batch.find_tensor_idx()
        batch[tensor_idx] = join(batch[tensor_idx], phony)

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        """Loads a skip tensor from the corresponding portal to pop. The given
        micro-batch is connected to the portal with :class:`Fork`.
        """
        if not self.skip_layout.requires_copy(ns, name):
            tensor = super().load(batch, ns, name)
            return tensor
        portal = self.portals[ns, name]
        tensor_idx = batch.find_tensor_idx()
        batch[tensor_idx], phony = fork(batch[tensor_idx])
        tensor = portal.orange(phony)
        return tensor

    def copy(self, batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream, ns: Namespace, name: str) -> None:
        """Copies the skip tensor in the corresponding portal. The given
        micro-batch and the portal will be tied with :class:`Fork` and
        :class:`Join`.
        """
        assert self.skip_layout.requires_copy(ns, name)
        tensor_idx = batch.find_tensor_idx()
        batch[tensor_idx], phony = fork(batch[tensor_idx])
        portal = self.portals[ns, name]
        phony = portal.copy(prev_stream, next_stream, phony)
        batch[tensor_idx] = join(batch[tensor_idx], phony)