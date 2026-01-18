from __future__ import annotations
import logging # isort:skip
import weakref
from typing import (
from tornado import gen
from ..application.application import ServerContext, SessionContext
from ..document import Document
from ..protocol.exceptions import ProtocolError
from ..util.token import get_token_payload
from .session import ServerSession
@property
def token_payload(self) -> TokenPayload:
    assert self._token is not None
    return get_token_payload(self._token)