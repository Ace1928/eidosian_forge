from __future__ import annotations
import functools
import warnings
from textwrap import dedent
from typing import Optional, Type, Union
def warning_format(message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str]=None) -> str:
    """
    Format for plotnine warnings
    """
    fmt = '{}:{}: {}: {}\n'.format
    return fmt(filename, lineno, category.__name__, message)