import os
from abc import ABC
from pathlib import Path
from typing import Any, Union
from .cloudpath import InvalidPrefixError, CloudPath
from .exceptions import AnyPathTypeError
def to_anypath(s: Union[str, os.PathLike]) -> Union[CloudPath, Path]:
    """Convenience method to convert a str or os.PathLike to the
    proper Path or CloudPath object using AnyPath.
    """
    if isinstance(s, (CloudPath, Path)):
        return s
    return AnyPath(s)