import abc
import os  # noqa
from typing import (
import attr
from . import hdrs
from .abc import AbstractView
from .typedefs import Handler, PathLike
class AbstractRouteDef(abc.ABC):

    @abc.abstractmethod
    def register(self, router: UrlDispatcher) -> List[AbstractRoute]:
        pass