from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
def _forceStopProcess(self, proc):
    """
        @param proc: An L{IProcessTransport} provider
        """
    try:
        proc.signalProcess('KILL')
    except error.ProcessExitedAlready:
        pass