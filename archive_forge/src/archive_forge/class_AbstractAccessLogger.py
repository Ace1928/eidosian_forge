import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sized
from http.cookies import BaseCookie, Morsel
from typing import (
from multidict import CIMultiDict
from yarl import URL
from .helpers import get_running_loop
from .typedefs import LooseCookies
class AbstractAccessLogger(ABC):
    """Abstract writer to access log."""

    def __init__(self, logger: logging.Logger, log_format: str) -> None:
        self.logger = logger
        self.log_format = log_format

    @abstractmethod
    def log(self, request: BaseRequest, response: StreamResponse, time: float) -> None:
        """Emit log to logger."""