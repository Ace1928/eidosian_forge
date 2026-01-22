import os
import uuid
import weakref
import collections
import functools
import numba
from numba.core import types, errors, utils, config
from numba.core.typing.typeof import typeof_impl  # noqa: F401
from numba.core.typing.asnumbatype import as_numba_type  # noqa: F401
from numba.core.typing.templates import infer, infer_getattr  # noqa: F401
from numba.core.imputils import (  # noqa: F401
from numba.core.datamodel import models   # noqa: F401
from numba.core.datamodel import register_default as register_model  # noqa: F401, E501
from numba.core.pythonapi import box, unbox, reflect, NativeValue  # noqa: F401
from numba._helperlib import _import_cython_function  # noqa: F401
from numba.core.serialize import ReduceMixin
class SentryLiteralArgs(collections.namedtuple('_SentryLiteralArgs', ['literal_args'])):
    """
    Parameters
    ----------
    literal_args : Sequence[str]
        A sequence of names for literal arguments

    Examples
    --------

    The following line:

    >>> SentryLiteralArgs(literal_args).for_pysig(pysig).bind(*args, **kwargs)

    is equivalent to:

    >>> sentry_literal_args(pysig, literal_args, args, kwargs)
    """

    def for_function(self, func):
        """Bind the sentry to the signature of *func*.

        Parameters
        ----------
        func : Function
            A python function.

        Returns
        -------
        obj : BoundLiteralArgs
        """
        return self.for_pysig(utils.pysignature(func))

    def for_pysig(self, pysig):
        """Bind the sentry to the given signature *pysig*.

        Parameters
        ----------
        pysig : inspect.Signature


        Returns
        -------
        obj : BoundLiteralArgs
        """
        return BoundLiteralArgs(pysig=pysig, literal_args=self.literal_args)