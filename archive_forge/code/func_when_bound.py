from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
def when_bound(self):
    if self.exchange:
        self.exchange = self.exchange(self.channel)