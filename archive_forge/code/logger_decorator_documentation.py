from functools import wraps
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from modin.config import LogMode
from .config import LogLevel, get_logger

            Compute function with logging if Modin logging is enabled.

            Parameters
            ----------
            *args : tuple
                The function arguments.
            **kwargs : dict
                The function keyword arguments.

            Returns
            -------
            Any
            