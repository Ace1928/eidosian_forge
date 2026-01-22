import logging
import os
from ovs import jsonrpc
from ovs import stream
from ovs import util as ovs_util
from ovs.db import schema
def _check_txn(self, error, reply):
    if error:
        ovs_util.ovs_fatal(error, os.strerror(error))
    elif reply.error:
        ovs_util.ovs_fatal(reply.error, 'error %s' % reply.error)