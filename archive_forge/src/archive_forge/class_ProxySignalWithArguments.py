import sys
import re
from .indenter import write_code
from .misc import Literal, moduleMember
class ProxySignalWithArguments(object):
    """ This is a proxy for (what should be) a signal that passes arguments.
    """

    def __init__(self, sender, signal_name, signal_index):
        self._sender = sender
        self._signal_name = signal_name
        if isinstance(signal_index, tuple):
            self._signal_index = ','.join(["'%s'" % a for a in signal_index])
        else:
            self._signal_index = "'%s'" % signal_index

    def connect(self, slot):
        write_code('%s.%s[%s].connect(%s) # type: ignore' % (self._sender, self._signal_name, self._signal_index, slot))