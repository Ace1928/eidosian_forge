import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
class ConchTestAvatar(avatar.ConchUser):
    """
    An avatar against which various SSH features can be tested.

    @ivar loggedOut: A flag indicating whether the avatar logout method has been
        called.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'
    loggedOut = False

    def __init__(self):
        avatar.ConchUser.__init__(self)
        self.listeners = {}
        self.globalRequests = {}
        self.channelLookup.update({b'session': session.SSHSession, b'direct-tcpip': forwarding.openConnectForwardingClient})
        self.subsystemLookup.update({b'crazy': CrazySubsystem})

    def global_foo(self, data):
        self.globalRequests['foo'] = data
        return 1

    def global_foo_2(self, data):
        self.globalRequests['foo_2'] = data
        return (1, b'data')

    def global_tcpip_forward(self, data):
        host, port = forwarding.unpackGlobal_tcpip_forward(data)
        try:
            listener = reactor.listenTCP(port, forwarding.SSHListenForwardingFactory(self.conn, (host, port), forwarding.SSHListenServerForwardingChannel), interface=host)
        except BaseException:
            log.err(None, 'something went wrong with remote->local forwarding')
            return 0
        else:
            self.listeners[host, port] = listener
            return 1

    def global_cancel_tcpip_forward(self, data):
        host, port = forwarding.unpackGlobal_tcpip_forward(data)
        listener = self.listeners.get((host, port), None)
        if not listener:
            return 0
        del self.listeners[host, port]
        listener.stopListening()
        return 1

    def logout(self):
        self.loggedOut = True
        for listener in self.listeners.values():
            log.msg('stopListening %s' % listener)
            listener.stopListening()