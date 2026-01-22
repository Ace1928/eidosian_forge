from __future__ import annotations
import logging # isort:skip
from ..application import SessionContext
from .lifecycle import LifecycleHandler
class DocumentLifecycleHandler(LifecycleHandler):
    """ Calls on_session_destroyed callbacks defined on the Document.
    """

    def __init__(self) -> None:
        super().__init__()
        self._on_session_destroyed = _on_session_destroyed