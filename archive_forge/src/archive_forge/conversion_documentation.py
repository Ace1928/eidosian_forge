import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface

        Create a Conversion context to use in a `with` statement.

        >>> with conversion_rules.context() as cv:
        ...     # Do something while using those conversion_rules.
        >>> # Do something else whith the earlier conversion rules restored.

        The conversion context is a *copy* of the converter object.

        :return: A :class:`ConversionContext`
        