import random
import threading
import time
from .messages import Message
from .parser import Parser
class BaseInput(BasePort):
    """Base class for input port.

    Subclass and override _receive() to create a new input port type.
    (See portmidi.py for an example of how to do this.)
    """
    is_input = True

    def __init__(self, name='', **kwargs):
        """Create an input port.

        name is the port name, as returned by input_names(). If
        name is not passed, the default input is used instead.
        """
        BasePort.__init__(self, name, **kwargs)
        self._parser = Parser()
        self._messages = self._parser.messages

    def _check_callback(self):
        if hasattr(self, 'callback') and self.callback is not None:
            raise ValueError('a callback is set for this port')

    def _receive(self, block=True):
        pass

    def iter_pending(self):
        """Iterate through pending messages."""
        while True:
            msg = self.poll()
            if msg is None:
                return
            else:
                yield msg

    def receive(self, block=True):
        """Return the next message.

        This will block until a message arrives.

        If you pass block=False it will not block and instead return
        None if there is no available message.

        If the port is closed and there are no pending messages IOError
        will be raised. If the port closes while waiting inside receive(),
        IOError will be raised. TODO: this seems a bit inconsistent. Should
        different errors be raised? What's most useful here?
        """
        if not self.is_input:
            raise ValueError('Not an input port')
        self._check_callback()
        with self._lock:
            if self._messages:
                return self._messages.popleft()
        if self.closed:
            if block:
                raise ValueError('receive() called on closed port')
            else:
                return None
        while True:
            with self._lock:
                msg = self._receive(block=block)
                if msg:
                    return msg
                if self._messages:
                    return self._messages.popleft()
                elif not block:
                    return None
                elif self.closed:
                    raise OSError('port closed during receive()')
            sleep()

    def poll(self):
        """Receive the next pending message or None

        This is the same as calling `receive(block=False)`."""
        return self.receive(block=False)

    def __iter__(self):
        """Iterate through messages until the port closes."""
        self._check_callback()
        while True:
            try:
                yield self.receive()
            except OSError:
                if self.closed:
                    return
                else:
                    raise