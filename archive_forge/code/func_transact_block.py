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
def transact_block(request, connection):
    """Emulate jsonrpc.Connection.transact_block without blocking eventlet.
    """
    error = connection.send(request)
    reply = None
    if error:
        return (error, reply)
    ovs_poller = poller.Poller()
    while not error:
        ovs_poller.immediate_wake()
        error, reply = connection.recv()
        if error != errno.EAGAIN:
            break
        if reply and reply.id == request.id and (reply.type in (jsonrpc.Message.T_REPLY, jsonrpc.Message.T_ERROR)):
            break
        connection.run()
        connection.wait(ovs_poller)
        connection.recv_wait(ovs_poller)
        ovs_poller.block()
        hub.sleep(0)
    return (error, reply)