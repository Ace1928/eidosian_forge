from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
def _create_bindings(self, nowait=False, channel=None):
    for B in self.bindings:
        channel = channel or self.channel
        B.declare(channel)
        B.bind(self, nowait=nowait, channel=channel)