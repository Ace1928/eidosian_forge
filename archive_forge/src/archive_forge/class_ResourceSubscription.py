import copy
import os
import sys
from io import BytesIO
from xml.dom.minidom import getDOMImplementation
from twisted.internet import address, reactor
from twisted.logger import Logger
from twisted.persisted import styles
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.web import http, resource, server, static, util
from twisted.web.http_headers import Headers
class ResourceSubscription(resource.Resource):
    isLeaf = 1
    waiting = 0
    _log = Logger()

    def __init__(self, host, port):
        resource.Resource.__init__(self)
        self.host = host
        self.port = port
        self.pending = []
        self.publisher = None

    def __getstate__(self):
        """Get persistent state for this ResourceSubscription."""
        state = copy.copy(self.__dict__)
        state['publisher'] = None
        state['waiting'] = 0
        state['pending'] = []
        return state

    def connected(self, publisher):
        """I've connected to a publisher; I'll now send all my requests."""
        self._log.info('connected to publisher')
        publisher.broker.notifyOnDisconnect(self.booted)
        self.publisher = publisher
        self.waiting = 0
        for request in self.pending:
            self.render(request)
        self.pending = []

    def notConnected(self, msg):
        """I can't connect to a publisher; I'll now reply to all pending
        requests.
        """
        self._log.info('could not connect to distributed web service: {msg}', msg=msg)
        self.waiting = 0
        self.publisher = None
        for request in self.pending:
            request.write('Unable to connect to distributed server.')
            request.finish()
        self.pending = []

    def booted(self):
        self.notConnected('connection dropped')

    def render(self, request):
        """Render this request, from my server.

        This will always be asynchronous, and therefore return NOT_DONE_YET.
        It spins off a request to the pb client, and either adds it to the list
        of pending issues or requests it immediately, depending on if the
        client is already connected.
        """
        if not self.publisher:
            self.pending.append(request)
            if not self.waiting:
                self.waiting = 1
                bf = pb.PBClientFactory()
                timeout = 10
                if self.host == 'unix':
                    reactor.connectUNIX(self.port, bf, timeout)
                else:
                    reactor.connectTCP(self.host, self.port, bf, timeout)
                d = bf.getRootObject()
                d.addCallbacks(self.connected, self.notConnected)
        else:
            i = Issue(request)
            self.publisher.callRemote('request', request).addCallbacks(i.finished, i.failed)
        return server.NOT_DONE_YET