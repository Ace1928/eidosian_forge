import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
class Win32Bus(wspbus.Bus):
    """A Web Site Process Bus implementation for Win32.

    Instead of time.sleep, this bus blocks using native win32event objects.
    """

    def __init__(self):
        self.events = {}
        wspbus.Bus.__init__(self)

    def _get_state_event(self, state):
        """Return a win32event for the given state (creating it if needed)."""
        try:
            return self.events[state]
        except KeyError:
            event = win32event.CreateEvent(None, 0, 0, 'WSPBus %s Event (pid=%r)' % (state.name, os.getpid()))
            self.events[state] = event
            return event

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        event = self._get_state_event(value)
        win32event.PulseEvent(event)

    def wait(self, state, interval=0.1, channel=None):
        """Wait for the given state(s), KeyboardInterrupt or SystemExit.

        Since this class uses native win32event objects, the interval
        argument is ignored.
        """
        if isinstance(state, (tuple, list)):
            if self.state not in state:
                events = tuple([self._get_state_event(s) for s in state])
                win32event.WaitForMultipleObjects(events, 0, win32event.INFINITE)
        elif self.state != state:
            event = self._get_state_event(state)
            win32event.WaitForSingleObject(event, win32event.INFINITE)