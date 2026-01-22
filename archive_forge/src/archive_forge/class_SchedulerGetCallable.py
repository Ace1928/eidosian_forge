from __future__ import annotations
import abc
from collections.abc import Callable, Hashable, Mapping, Sequence
from enum import Enum
from typing import (
class SchedulerGetCallable(Protocol):
    """Protocol defining the signature of a ``__dask_scheduler__`` callable."""

    def __call__(self, dsk: Graph, keys: Sequence[Key] | Key, **kwargs: Any) -> Any:
        """Method called as the default scheduler for a collection.

        Parameters
        ----------
        dsk :
            The task graph.
        keys :
            Key(s) corresponding to the desired data.
        **kwargs :
            Additional arguments.

        Returns
        -------
        Any
            Result(s) associated with `keys`

        """
        raise NotImplementedError('Inheriting class must implement this method.')