from __future__ import annotations
import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
from .labels import (
from .utils import identity
class trans(ABC):
    """
    Base class for all transforms

    This class is used to transform data and also tell the
    x and y axes how to create and label the tick locations.

    The key methods to override are :meth:`trans.transform`
    and :meth:`trans.inverse`. Alternately, you can quickly
    create a transform class using the :func:`trans_new`
    function.

    Parameters
    ----------
    kwargs : dict
        Attributes of the class to set/override

    """
    transform_is_linear: bool = False
    domain: DomainType = (-np.inf, np.inf)
    breaks_: BreaksFunction = breaks_extended(n=5)
    format: FormatFunction = staticmethod(label_number())

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f'Unknown Parameter: {k}')

    @property
    def domain_is_numerical(self) -> bool:
        """
        Return True if transformation acts on numerical data.
        e.g. int, float, and imag are numerical but datetime
        is not.

        """
        return isinstance(self.domain[0], (int, float, np.number))

    def minor_breaks(self, major: FloatArrayLike, limits: Optional[TupleFloat2]=None, n: Optional[int]=None) -> NDArrayFloat:
        """
        Calculate minor_breaks
        """
        n = 1 if n is None else n
        if self.transform_is_linear or not self.domain_is_numerical:
            func = minor_breaks(n=n)
        else:
            func = minor_breaks_trans(self, n=n)
        return func(major, limits, n)

    @staticmethod
    @abstractmethod
    def transform(x: TFloatArrayLike) -> TFloatArrayLike:
        """
        Transform of x
        """
        ...

    @staticmethod
    @abstractmethod
    def inverse(x: TFloatArrayLike) -> TFloatArrayLike:
        """
        Inverse of x
        """
        ...

    def breaks(self, limits: tuple[Any, Any]) -> NDArrayAny:
        """
        Calculate breaks in data space and return them
        in transformed space.

        Expects limits to be in *transform space*, this
        is the same space as that where the domain is
        specified.

        This method wraps around :meth:`breaks_` to ensure
        that the calculated breaks are within the domain
        the transform. This is helpful in cases where an
        aesthetic requests breaks with limits expanded for
        some padding, yet the expansion goes beyond the
        domain of the transform. e.g for a probability
        transform the breaks will be in the domain
        ``[0, 1]`` despite any outward limits.

        Parameters
        ----------
        limits : tuple
            The scale limits. Size 2.

        Returns
        -------
        out : array_like
            Major breaks
        """
        limits = (max(self.domain[0], limits[0]), min(self.domain[1], limits[1]))
        breaks = np.asarray(self.breaks_(limits))
        breaks = breaks.compress((breaks >= self.domain[0]) & (breaks <= self.domain[1]))
        return breaks