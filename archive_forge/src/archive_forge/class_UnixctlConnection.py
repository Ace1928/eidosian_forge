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
class UnixctlConnection(object):

    def __init__(self, rpc):
        assert isinstance(rpc, ovs.jsonrpc.Connection)
        self._rpc = rpc
        self._request_id = None

    def run(self):
        self._rpc.run()
        error = self._rpc.get_status()
        if error or self._rpc.get_backlog():
            return error
        for _ in range(10):
            if error or self._request_id:
                break
            error, msg = self._rpc.recv()
            if msg:
                if msg.type == Message.T_REQUEST:
                    self._process_command(msg)
                else:
                    vlog.warn('%s: received unexpected %s message' % (self._rpc.name, Message.type_to_string(msg.type)))
                    error = errno.EINVAL
            if not error:
                error = self._rpc.get_status()
        return error

    def reply(self, body):
        self._reply_impl(True, body)

    def reply_error(self, body):
        self._reply_impl(False, body)

    def _close(self):
        self._rpc.close()
        self._request_id = None

    def _wait(self, poller):
        self._rpc.wait(poller)
        if not self._rpc.get_backlog():
            self._rpc.recv_wait(poller)

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

    def _process_command(self, request):
        assert isinstance(request, ovs.jsonrpc.Message)
        assert request.type == ovs.jsonrpc.Message.T_REQUEST
        self._request_id = request.id
        error = None
        params = request.params
        method = request.method
        command = ovs.unixctl.commands.get(method)
        if command is None:
            error = '"%s" is not a valid command' % method
        elif len(params) < command.min_args:
            error = '"%s" command requires at least %d arguments' % (method, command.min_args)
        elif len(params) > command.max_args:
            error = '"%s" command takes at most %d arguments' % (method, command.max_args)
        else:
            for param in params:
                if not isinstance(param, str):
                    error = '"%s" command has non-string argument' % method
                    break
            if error is None:
                unicode_params = [str(p) for p in params]
                command.callback(self, unicode_params, command.aux)
        if error:
            self.reply_error(error)