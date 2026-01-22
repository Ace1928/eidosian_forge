from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
class ServerContext(metaclass=ABCMeta):
    """ A harness for server-specific information and tasks related to
    collections of Bokeh sessions.

    *This base class is probably not of interest to general users.*

    """

    @property
    @abstractmethod
    def sessions(self) -> list[ServerSession]:
        """ ``SessionContext`` instances belonging to this application.

        *Subclasses must implement this method.*

        """
        pass