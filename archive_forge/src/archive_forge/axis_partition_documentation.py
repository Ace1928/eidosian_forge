from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from modin.logging import ClassLogger

        Unwrap partitions from this axis partition.

        Parameters
        ----------
        squeeze : bool, default: False
            Flag used to unwrap only one partition.
        get_ip : bool, default: False
            Whether to get node ip address to each partition or not.

        Returns
        -------
        list
            List of partitions from this axis partition.

        Notes
        -----
        If `get_ip=True`, a tuple of lists of Ray.ObjectRef/Dask.Future to node ip addresses and
        unwrapped partitions, respectively, is returned if Ray/Dask is used as an engine
        (i.e. [(Ray.ObjectRef/Dask.Future, Ray.ObjectRef/Dask.Future), ...]).
        