from __future__ import annotations
import logging # isort:skip
import inspect
import time
from copy import copy
from functools import wraps
from typing import (
from tornado import locks
from ..events import ConnectionLost
from ..util.token import generate_jwt_token
from .callbacks import DocumentCallbackGroup
class ServerSession:
    """ Hosts an application "instance" (an instantiated Document) for one or more connections.

    """
    _subscribed_connections: set[ServerConnection]
    _current_patch_connection: ServerConnection | None
    _pending_writes: list[Awaitable[None]] | None

    def __init__(self, session_id: ID, document: Document, io_loop: IOLoop | None=None, token: str | None=None) -> None:
        if session_id is None:
            raise ValueError('Sessions must have an id')
        if document is None:
            raise ValueError('Sessions must have a document')
        self._id = session_id
        self._token = token
        self._document = document
        self._loop = io_loop
        self._subscribed_connections = set()
        self._last_unsubscribe_time = current_time()
        self._lock = locks.Lock()
        self._current_patch_connection = None
        self._document.callbacks.on_change_dispatch_to(self)
        self._callbacks = DocumentCallbackGroup(io_loop)
        self._pending_writes = None
        self._destroyed = False
        self._expiration_requested = False
        self._expiration_blocked_count = 0
        wrapped_callbacks = [self._wrap_session_callback(cb) for cb in self._document.session_callbacks]
        self._callbacks.add_session_callbacks(wrapped_callbacks)

    @property
    def document(self) -> Document:
        return self._document

    @property
    def id(self) -> ID:
        return self._id

    @property
    def token(self) -> str:
        """ A JWT token to authenticate the session. """
        if self._token:
            return self._token
        return generate_jwt_token(self.id)

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def expiration_requested(self) -> bool:
        return self._expiration_requested

    @property
    def expiration_blocked(self) -> bool:
        return self._expiration_blocked_count > 0

    @property
    def expiration_blocked_count(self) -> int:
        return self._expiration_blocked_count

    def destroy(self) -> None:
        self._destroyed = True
        self._document.destroy(self)
        del self._document
        self._callbacks.remove_all_callbacks()
        del self._callbacks

    def request_expiration(self) -> None:
        """ Used in test suite for now. Forces immediate expiration if no connections."""
        self._expiration_requested = True

    def block_expiration(self) -> None:
        self._expiration_blocked_count += 1

    def unblock_expiration(self) -> None:
        if self._expiration_blocked_count <= 0:
            raise RuntimeError('mismatched block_expiration / unblock_expiration')
        self._expiration_blocked_count -= 1

    def subscribe(self, connection: ServerConnection) -> None:
        """This should only be called by ``ServerConnection.subscribe_session`` or our book-keeping will be broken"""
        self._subscribed_connections.add(connection)

    def unsubscribe(self, connection: ServerConnection) -> None:
        """This should only be called by ``ServerConnection.unsubscribe_session`` or our book-keeping will be broken"""
        self._subscribed_connections.discard(connection)
        self._last_unsubscribe_time = current_time()

    @property
    def connection_count(self) -> int:
        return len(self._subscribed_connections)

    @property
    def milliseconds_since_last_unsubscribe(self) -> float:
        return current_time() - self._last_unsubscribe_time

    @_needs_document_lock
    def with_document_locked(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """ Asynchronously locks the document and runs the function with it locked."""
        return func(*args, **kwargs)

    def _wrap_document_callback(self, callback: Callback) -> Callback:
        if getattr(callback, 'nolock', False):
            return callback

        def wrapped_callback(*args: Any, **kwargs: Any):
            return self.with_document_locked(callback, *args, **kwargs)
        return wrapped_callback

    def _wrap_session_callback(self, callback: SessionCallback) -> SessionCallback:
        wrapped = copy(callback)
        wrapped._callback = self._wrap_document_callback(callback.callback)
        return wrapped

    def _document_patched(self, event: DocumentPatchedEvent) -> None:
        may_suppress = event.setter is self
        if self._pending_writes is None:
            raise RuntimeError('_pending_writes should be non-None when we have a document lock, and we should have the lock when the document changes')
        for connection in self._subscribed_connections:
            if may_suppress and connection is self._current_patch_connection:
                continue
            self._pending_writes.append(connection.send_patch_document(event))

    @_needs_document_lock
    def _handle_pull(self, message: msg.pull_doc_req, connection: ServerConnection) -> msg.pull_doc_reply:
        log.debug(f'Sending pull-doc-reply from session {self.id!r}')
        return connection.protocol.create('PULL-DOC-REPLY', message.header['msgid'], self.document)

    def _session_callback_added(self, event: SessionCallback):
        wrapped = self._wrap_session_callback(event.callback)
        self._callbacks.add_session_callback(wrapped)

    def _session_callback_removed(self, event):
        self._callbacks.remove_session_callback(event.callback)

    @classmethod
    def pull(cls, message: msg.pull_doc_req, connection: ServerConnection) -> msg.pull_doc_reply:
        """ Handle a PULL-DOC, return a Future with work to be scheduled. """
        return connection.session._handle_pull(message, connection)

    @_needs_document_lock
    def _handle_push(self, message: msg.push_doc, connection: ServerConnection) -> msg.ok:
        log.debug(f'pushing doc to session {self.id!r}')
        message.push_to_document(self.document)
        return connection.ok(message)

    @classmethod
    def push(cls, message: msg.push_doc, connection: ServerConnection) -> msg.ok:
        """ Handle a PUSH-DOC, return a Future with work to be scheduled. """
        return connection.session._handle_push(message, connection)

    @_needs_document_lock
    def _handle_patch(self, message: msg.patch_doc, connection: ServerConnection) -> msg.ok:
        self._current_patch_connection = connection
        try:
            message.apply_to_document(self.document, self)
        finally:
            self._current_patch_connection = None
        return connection.ok(message)

    @classmethod
    def patch(cls, message: msg.patch_doc, connection: ServerConnection) -> msg.ok:
        """ Handle a PATCH-DOC, return a Future with work to be scheduled. """
        return connection.session._handle_patch(message, connection)

    def notify_connection_lost(self) -> None:
        """ Notify the document that the connection was lost. """
        self.document.callbacks.trigger_event(ConnectionLost())