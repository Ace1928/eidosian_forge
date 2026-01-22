import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
def _check_valid_iterator_id(self, iterator_id) -> bool:
    """Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`."""
    return iterator_id == self._valid_iterator_id and iterator_id == self.main_datapipe._valid_iterator_id