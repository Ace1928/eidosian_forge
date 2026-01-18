from __future__ import annotations
import logging # isort:skip
from typing import Any, TypedDict
from bokeh import __version__
from ...core.types import ID
from ..message import Message
 Create an ``SERVER-INFO-REPLY`` message

        Args:
            request_id (str) :
                The message ID for the message that issues the info request

        Any additional keyword arguments will be put into the message
        ``metadata`` fragment as-is.

        