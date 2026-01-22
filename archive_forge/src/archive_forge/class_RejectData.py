from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Sequence, TypeVar, Union
from .extensions import Extension
from .typing import Headers
@dataclass(frozen=True)
class RejectData(Event):
    """The rejection HTTP response body.

    The caller may send multiple ``RejectData`` events. The final event should
    have the ``body_finished`` attribute set to ``True``.

    Fields:

    .. attribute:: body_finished

       True if this is the final chunk of the body data.

    .. attribute:: data (bytes)

       (Required) The raw body data.

    """
    data: bytes
    body_finished: bool = True