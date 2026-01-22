import typing
from sys import stderr, stdout
from textwrap import dedent
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, cast
from twisted.copyright import version
from twisted.internet.interfaces import IReactorCore
from twisted.logger import (
from twisted.plugin import getPlugins
from twisted.python.usage import Options, UsageError
from ..reactors import NoSuchReactor, getReactorTypes, installReactor
from ..runner._exit import ExitStatus, exit
from ..service import IServiceMaker
def _update_doc(opt: Callable[['TwistOptions', str], None], **kwargs: str) -> None:
    """
    Update the docstring of a method that implements an option.
    The string is dedented and the given keyword arguments are substituted.
    """
    opt.__doc__ = dedent(opt.__doc__ or '').format(**kwargs)