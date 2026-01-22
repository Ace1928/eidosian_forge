from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorWin32Events(Interface):
    """
    Win32 Event API methods

    @since: 10.2
    """

    def addEvent(event: object, fd: 'FileDescriptor', action: str) -> None:
        """
        Add a new win32 event to the event loop.

        @param event: a Win32 event object created using win32event.CreateEvent()
        @param fd: an instance of L{twisted.internet.abstract.FileDescriptor}
        @param action: a string that is a method name of the fd instance.
                       This method is called in response to the event.
        """

    def removeEvent(event: object) -> None:
        """
        Remove an event.

        @param event: a Win32 event object added using L{IReactorWin32Events.addEvent}

        @return: None
        """