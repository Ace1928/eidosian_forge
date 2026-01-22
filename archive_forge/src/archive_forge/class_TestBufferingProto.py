from __future__ import annotations
from twisted.conch import mixin
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
class TestBufferingProto(mixin.BufferingMixin):
    scheduled = False
    rescheduled = 0
    transport: StringTransport

    def schedule(self) -> object:
        self.scheduled = True
        return object()

    def reschedule(self, token: object) -> None:
        self.rescheduled += 1