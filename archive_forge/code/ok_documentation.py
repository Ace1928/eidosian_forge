from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.types import ID
from ..message import Empty, Message
 Create an ``OK`` message

        Args:
            request_id (str) :
                The message ID for the message the precipitated the OK.

        Any additional keyword arguments will be put into the message
        ``metadata`` fragment as-is.

        