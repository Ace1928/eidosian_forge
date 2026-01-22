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
def _handle_pull(self, message: msg.pull_doc_req, connection: ServerConnection) -> msg.pull_doc_reply:
    log.debug(f'Sending pull-doc-reply from session {self.id!r}')
    return connection.protocol.create('PULL-DOC-REPLY', message.header['msgid'], self.document)