from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IPullProducer(IProducer):
    """
    A pull producer, also known as a non-streaming producer, is
    expected to produce data each time L{resumeProducing()} is called.
    """

    def resumeProducing() -> None:
        """
        Produce data for the consumer a single time.

        This tells a producer to produce data for the consumer once
        (not repeatedly, once only). Typically this will be done
        by calling the consumer's C{write} method a single time with
        produced data. The producer should produce data before returning
        from C{resumeProducing()}, that is, it should not schedule a deferred
        write.
        """