import asyncio
import sys
from asyncio import AbstractEventLoop, AbstractEventLoopPolicy
from contextlib import suppress
from typing import Any, Callable, Dict, Optional, Sequence, Type
from warnings import catch_warnings, filterwarnings, warn
from twisted.internet import asyncioreactor, error
from twisted.internet.base import DelayedCall
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.misc import load_object
def verify_installed_reactor(reactor_path: str) -> None:
    """Raises :exc:`Exception` if the installed
    :mod:`~twisted.internet.reactor` does not match the specified import
    path."""
    from twisted.internet import reactor
    reactor_class = load_object(reactor_path)
    if not reactor.__class__ == reactor_class:
        msg = f'The installed reactor ({reactor.__module__}.{reactor.__class__.__name__}) does not match the requested one ({reactor_path})'
        raise Exception(msg)