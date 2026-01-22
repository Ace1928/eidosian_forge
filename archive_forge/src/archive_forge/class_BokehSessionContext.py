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
class BokehSessionContext(SessionContext):
    _session: ServerSession | None
    _request: _RequestProxy | None
    _token: str | None

    def __init__(self, session_id: ID, server_context: ServerContext, document: Document, logout_url: str | None=None) -> None:
        self._document = document
        self._session = None
        self._logout_url = logout_url
        super().__init__(server_context, session_id)
        self._request = None
        self._token = None

    def _set_session(self, session: ServerSession) -> None:
        self._session = session

    async def with_locked_document(self, func: Callable[[Document], Awaitable[None]]) -> None:
        if self._session is None:
            await func(self._document)
        else:
            await self._session.with_document_locked(func, self._document)

    @property
    def destroyed(self) -> bool:
        if self._session is None:
            return False
        else:
            return self._session.destroyed

    @property
    def logout_url(self) -> str | None:
        return self._logout_url

    @property
    def request(self) -> _RequestProxy | None:
        return self._request

    @property
    def token_payload(self) -> TokenPayload:
        assert self._token is not None
        return get_token_payload(self._token)

    @property
    def session(self) -> ServerSession | None:
        return self._session