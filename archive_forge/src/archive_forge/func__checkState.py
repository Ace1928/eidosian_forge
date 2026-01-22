import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
def _checkState(manager):
    """
    Prompt a relaying manager to check state.

    @type manager: L{SmartHostSMTPRelayingManager}
    @param manager: A relaying manager.
    """
    manager.checkState()