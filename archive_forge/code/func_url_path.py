from __future__ import annotations
import logging # isort:skip
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any
from ...document import Document
from ..application import ServerContext, SessionContext
def url_path(self) -> str | None:
    """ Returns a default URL path, if applicable.

        Handlers subclasses may optionally implement this method, to inform
        the Bokeh application what URL it should be installed at.

        If multiple handlers specify ``url_path`` the Application will use the
        value from the first handler in its list of handlers.

        """
    return None