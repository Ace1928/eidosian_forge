import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
def _set_main_datapipe_valid_iterator_id(self) -> int:
    """
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.

        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
    if self.main_datapipe._valid_iterator_id is None:
        self.main_datapipe._valid_iterator_id = 0
    elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:
        self.main_datapipe._valid_iterator_id += 1
        if not self.main_datapipe.is_every_instance_exhausted():
            warnings.warn('Some child DataPipes are not exhausted when __iter__ is called. We are resetting the buffer and each child DataPipe will read from the start again.', UserWarning)
        self.main_datapipe.reset()
    self._valid_iterator_id = self.main_datapipe._valid_iterator_id
    return self._valid_iterator_id