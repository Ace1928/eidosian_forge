import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
def _idl_loop(self):
    while self.is_active:
        try:
            self._idl.run()
            self._transactions()
        except Exception:
            self.logger.exception('Error running IDL for system_id %s' % self.system_id)
            raise
        hub.sleep(0)