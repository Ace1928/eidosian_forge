from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
def upgrateToVersion2(self):
    for k in ('_groups', '_persons'):
        if not hasattr(self, k):
            setattr(self, k, {})