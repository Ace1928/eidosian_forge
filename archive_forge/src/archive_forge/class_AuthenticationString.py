import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
class AuthenticationString(bytes):

    def __reduce__(self):
        from .context import get_spawning_popen
        if get_spawning_popen() is None:
            raise TypeError('Pickling an AuthenticationString object is disallowed for security reasons')
        return (AuthenticationString, (bytes(self),))