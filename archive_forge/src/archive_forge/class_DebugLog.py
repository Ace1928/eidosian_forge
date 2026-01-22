import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
class DebugLog(StaticClass):
    """Static functions for debug logging."""

    @staticmethod
    def log_text(text):
        """
        Log lines of text, inserting a timestamp.

        @type  text: str
        @param text: Text to log.

        @rtype:  str
        @return: Log line.
        """
        if text.endswith('\n'):
            text = text[:-len('\n')]
        ltime = time.strftime('%X')
        msecs = time.time() % 1 * 1000
        return '[%s.%04d] %s' % (ltime, msecs, text)

    @classmethod
    def log_event(cls, event, text=None):
        """
        Log lines of text associated with a debug event.

        @type  event: L{Event}
        @param event: Event object.

        @type  text: str
        @param text: (Optional) Text to log. If no text is provided the default
            is to show a description of the event itself.

        @rtype:  str
        @return: Log line.
        """
        if not text:
            if event.get_event_code() == win32.EXCEPTION_DEBUG_EVENT:
                what = event.get_exception_description()
                if event.is_first_chance():
                    what = '%s (first chance)' % what
                else:
                    what = '%s (second chance)' % what
                try:
                    address = event.get_fault_address()
                except NotImplementedError:
                    address = event.get_exception_address()
            else:
                what = event.get_event_name()
                address = event.get_thread().get_pc()
            process = event.get_process()
            label = process.get_label_at_address(address)
            address = HexDump.address(address, process.get_bits())
            if label:
                where = '%s (%s)' % (address, label)
            else:
                where = address
            text = '%s at %s' % (what, where)
        text = 'pid %d tid %d: %s' % (event.get_pid(), event.get_tid(), text)
        return cls.log_text(text)