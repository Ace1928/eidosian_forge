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
class BokehServerContext(ServerContext):

    def __init__(self, application_context: ApplicationContext) -> None:
        self._application_context = weakref.ref(application_context)

    @property
    def application_context(self) -> ApplicationContext | None:
        return self._application_context()

    @property
    def sessions(self) -> list[ServerSession]:
        result: list[ServerSession] = []
        context = self.application_context
        if context:
            for session in context.sessions:
                result.append(session)
        return result