import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
class Joinable(ABC):
    """
    This defines an abstract base class for joinable classes.

    A joinable class
    (inheriting from :class:`Joinable`) should implement :meth:`join_hook`,
    which returns a :class:`JoinHook` instance, in addition to
    :meth:`join_device` and :meth:`join_process_group` that return device and
    process group information, respectively.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._join_config = _JoinConfig.construct_disabled_join_config()

    @abstractmethod
    def join_hook(self, **kwargs) -> JoinHook:
        """
        Return a :class:`JoinHook` instance for the given :class:`Joinable`.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.
        """
        ...

    @property
    @abstractmethod
    def join_device(self) -> torch.device:
        """Return the device from which to perform collective communications needed by the join context manager."""
        ...

    @property
    @abstractmethod
    def join_process_group(self) -> Any:
        """Returns the process group for the collective communications needed by the join context manager itself."""
        ...