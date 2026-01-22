from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, TypedDict
from ..exceptions import ProtocolError
from ..message import Message
class PushDoc(TypedDict):
    doc: DocJson