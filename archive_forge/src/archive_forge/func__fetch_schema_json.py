import logging
import os
from ovs import jsonrpc
from ovs import stream
from ovs import util as ovs_util
from ovs.db import schema
def _fetch_schema_json(self, rpc, database):
    request = jsonrpc.Message.create_request('get_schema', [database])
    error, reply = rpc.transact_block(request)
    self._check_txn(error, reply)
    return reply.result