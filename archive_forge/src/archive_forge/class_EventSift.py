from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
class EventSift(EventHandler):
    """
    Event handler that allows you to use customized event handlers for each
    process you're attached to.

    This makes coding the event handlers much easier, because each instance
    will only "know" about one process. So you can code your event handler as
    if only one process was being debugged, but your debugger can attach to
    multiple processes.

    Example::
        from winappdbg import Debug, EventHandler, EventSift

        # This class was written assuming only one process is attached.
        # If you used it directly it would break when attaching to another
        # process, or when a child process is spawned.
        class MyEventHandler (EventHandler):

            def create_process(self, event):
                self.first = True
                self.name = event.get_process().get_filename()
                print "Attached to %s" % self.name

            def breakpoint(self, event):
                if self.first:
                    self.first = False
                    print "First breakpoint reached at %s" % self.name

            def exit_process(self, event):
                print "Detached from %s" % self.name

        # Now when debugging we use the EventSift to be able to work with
        # multiple processes while keeping our code simple. :)
        if __name__ == "__main__":
            handler = EventSift(MyEventHandler)
            #handler = MyEventHandler()  # try uncommenting this line...
            with Debug(handler) as debug:
                debug.execl("calc.exe")
                debug.execl("notepad.exe")
                debug.execl("charmap.exe")
                debug.loop()

    Subclasses of C{EventSift} can prevent specific event types from
    being forwarded by simply defining a method for it. That means your
    subclass can handle some event types globally while letting other types
    be handled on per-process basis. To forward events manually you can
    call C{self.event(event)}.

    Example::
        class MySift (EventSift):

            # Don't forward this event.
            def debug_control_c(self, event):
                pass

            # Handle this event globally without forwarding it.
            def output_string(self, event):
                print "Debug string: %s" % event.get_debug_string()

            # Handle this event globally and then forward it.
            def create_process(self, event):
                print "New process created, PID: %d" % event.get_pid()
                return self.event(event)

            # All other events will be forwarded.

    Note that overriding the C{event} method would cause no events to be
    forwarded at all. To prevent this, call the superclass implementation.

    Example::

        def we_want_to_forward_this_event(event):
            "Use whatever logic you want here..."
            # (...return True or False...)

        class MySift (EventSift):

            def event(self, event):

                # If the event matches some custom criteria...
                if we_want_to_forward_this_event(event):

                    # Forward it.
                    return super(MySift, self).event(event)

                # Otherwise, don't.

    @type cls: class
    @ivar cls:
        Event handler class. There will be one instance of this class
        per debugged process in the L{forward} dictionary.

    @type argv: list
    @ivar argv:
        Positional arguments to pass to the constructor of L{cls}.

    @type argd: list
    @ivar argd:
        Keyword arguments to pass to the constructor of L{cls}.

    @type forward: dict
    @ivar forward:
        Dictionary that maps each debugged process ID to an instance of L{cls}.
    """

    def __init__(self, cls, *argv, **argd):
        """
        Maintains an instance of your event handler for each process being
        debugged, and forwards the events of each process to each corresponding
        instance.

        @warn: If you subclass L{EventSift} and reimplement this method,
            don't forget to call the superclass constructor!

        @see: L{event}

        @type  cls: class
        @param cls: Event handler class. This must be the class itself, not an
            instance! All additional arguments passed to the constructor of
            the event forwarder will be passed on to the constructor of this
            class as well.
        """
        self.cls = cls
        self.argv = argv
        self.argd = argd
        self.forward = dict()
        super(EventSift, self).__init__()

    def __call__(self, event):
        try:
            eventCode = event.get_event_code()
            if eventCode in (win32.LOAD_DLL_DEBUG_EVENT, win32.LOAD_DLL_DEBUG_EVENT):
                pid = event.get_pid()
                handler = self.forward.get(pid, None)
                if handler is None:
                    handler = self.cls(*self.argv, **self.argd)
                self.forward[pid] = handler
                if isinstance(handler, EventHandler):
                    if eventCode == win32.LOAD_DLL_DEBUG_EVENT:
                        handler.__EventHandler_hook_dll(event)
                    else:
                        handler.__EventHandler_unhook_dll(event)
        finally:
            return super(EventSift, self).__call__(event)

    def event(self, event):
        """
        Forwards events to the corresponding instance of your event handler
        for this process.

        If you subclass L{EventSift} and reimplement this method, no event
        will be forwarded at all unless you call the superclass implementation.

        If your filtering is based on the event type, there's a much easier way
        to do it: just implement a handler for it.
        """
        eventCode = event.get_event_code()
        pid = event.get_pid()
        handler = self.forward.get(pid, None)
        if handler is None:
            handler = self.cls(*self.argv, **self.argd)
            if eventCode != win32.EXIT_PROCESS_DEBUG_EVENT:
                self.forward[pid] = handler
        elif eventCode == win32.EXIT_PROCESS_DEBUG_EVENT:
            del self.forward[pid]
        return handler(event)