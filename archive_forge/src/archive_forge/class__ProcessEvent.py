import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class _ProcessEvent:
    """
    Abstract processing event class.
    """

    def __call__(self, event):
        """
        To behave like a functor the object must be callable.
        This method is a dispatch method. Its lookup order is:
          1. process_MASKNAME method
          2. process_FAMILY_NAME method
          3. otherwise calls process_default

        @param event: Event to be processed.
        @type event: Event object
        @return: By convention when used from the ProcessEvent class:
                 - Returning False or None (default value) means keep on
                 executing next chained functors (see chain.py example).
                 - Returning True instead means do not execute next
                   processing functions.
        @rtype: bool
        @raise ProcessEventError: Event object undispatchable,
                                  unknown event.
        """
        stripped_mask = event.mask - (event.mask & IN_ISDIR)
        maskname = EventsCodes.ALL_VALUES.get(stripped_mask)
        if maskname is None:
            raise ProcessEventError('Unknown mask 0x%08x' % stripped_mask)
        meth = getattr(self, 'process_' + maskname, None)
        if meth is not None:
            return meth(event)
        meth = getattr(self, 'process_IN_' + maskname.split('_')[1], None)
        if meth is not None:
            return meth(event)
        return self.process_default(event)

    def __repr__(self):
        return '<%s>' % self.__class__.__name__