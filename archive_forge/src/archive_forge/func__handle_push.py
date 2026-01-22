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
@_needs_document_lock
def _handle_push(self, message: msg.push_doc, connection: ServerConnection) -> msg.ok:
    log.debug(f'pushing doc to session {self.id!r}')
    message.push_to_document(self.document)
    return connection.ok(message)