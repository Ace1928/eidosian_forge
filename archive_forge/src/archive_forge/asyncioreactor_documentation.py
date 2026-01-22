import errno
import sys
from asyncio import AbstractEventLoop, get_event_loop
from typing import Dict, Optional, Type
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import (
from twisted.logger import Logger
from twisted.python.log import callWithLogger

        Compensate for a bug in asyncio where it will not unregister a FD that
        it cannot handle in the epoll loop. It touches internal asyncio code.

        A description of the bug by markrwilliams:

        The C{add_writer} method of asyncio event loops isn't atomic because
        all the Selector classes in the selector module internally record a
        file object before passing it to the platform's selector
        implementation. If the platform's selector decides the file object
        isn't acceptable, the resulting exception doesn't cause the Selector to
        un-track the file object.

        The failing/hanging stdio test goes through the following sequence of
        events (roughly):

        * The first C{connection.write(intToByte(value))} call hits the asyncio
        reactor's C{addWriter} method.

        * C{addWriter} calls the asyncio loop's C{add_writer} method, which
        happens to live on C{_BaseSelectorEventLoop}.

        * The asyncio loop's C{add_writer} method checks if the file object has
        been registered before via the selector's C{get_key} method.

        * It hasn't, so the KeyError block runs and calls the selector's
        register method

        * Code examples that follow use EpollSelector, but the code flow holds
        true for any other selector implementation. The selector's register
        method first calls through to the next register method in the MRO

        * That next method is always C{_BaseSelectorImpl.register} which
        creates a C{SelectorKey} instance for the file object, stores it under
        the file object's file descriptor, and then returns it.

        * Control returns to the concrete selector implementation, which asks
        the operating system to track the file descriptor using the right API.

        * The operating system refuses! An exception is raised that, in this
        case, the asyncio reactor handles by creating a C{_ContinuousPolling}
        object to watch the file descriptor.

        * The second C{connection.write(intToByte(value))} call hits the
        asyncio reactor's C{addWriter} method, which hits the C{add_writer}
        method. But the loop's selector's get_key method now returns a
        C{SelectorKey}! Now the asyncio reactor's C{addWriter} method thinks
        the asyncio loop will watch the file descriptor, even though it won't.
        