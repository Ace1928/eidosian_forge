import copy
import errno
import os
import sys
import ovs.dirs
import ovs.jsonrpc
import ovs.stream
import ovs.unixctl
import ovs.util
import ovs.version
import ovs.vlog
def _reply_impl(self, success, body):
    assert isinstance(success, bool)
    assert body is None or isinstance(body, str)
    assert self._request_id is not None
    if body is None:
        body = ''
    if body and (not body.endswith('\n')):
        body += '\n'
    if success:
        reply = Message.create_reply(body, self._request_id)
    else:
        reply = Message.create_error(body, self._request_id)
    self._rpc.send(reply)
    self._request_id = None