import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def _parse_userauth_failure(self, m):
    authlist = m.get_list()
    partial = m.get_boolean()
    if partial:
        self._log(INFO, 'Authentication continues...')
        self._log(DEBUG, 'Methods: ' + str(authlist))
        self.transport.saved_exception = PartialAuthentication(authlist)
    elif self.auth_method not in authlist:
        for msg in ('Authentication type ({}) not permitted.'.format(self.auth_method), 'Allowed methods: {}'.format(authlist)):
            self._log(DEBUG, msg)
        self.transport.saved_exception = BadAuthenticationType('Bad authentication type', authlist)
    else:
        self._log(INFO, 'Authentication ({}) failed.'.format(self.auth_method))
    self.authenticated = False
    self.username = None
    if self.auth_event is not None:
        self.auth_event.set()