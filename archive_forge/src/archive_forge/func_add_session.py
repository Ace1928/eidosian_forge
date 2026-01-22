import os
import time
from threading import Thread, Lock
from contextlib import contextmanager
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.session import Session
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
def add_session(self, session):
    if session.session_mode == 'request':
        self.add_aggregate_session(session)
    else:
        self.pending_sessions.append(session.to_json())
    self._ensure_running()