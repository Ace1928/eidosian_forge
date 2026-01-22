from functools import wraps
import weakref
import abc
import warnings
from ..data_sparsifier import BaseDataSparsifier
Loads the schedulers state.

        Note:
            Remember to restore the state of the data_sparsifier before the scheduler.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        