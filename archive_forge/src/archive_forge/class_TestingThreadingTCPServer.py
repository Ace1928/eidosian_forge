import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestingThreadingTCPServer(TestingTCPServerMixin, socketserver.ThreadingTCPServer):

    def __init__(self, server_address, request_handler_class):
        TestingTCPServerMixin.__init__(self)
        socketserver.ThreadingTCPServer.__init__(self, server_address, request_handler_class)

    def get_request(self):
        """Get the request and client address from the socket."""
        sock, addr = TestingTCPServerMixin.get_request(self)
        self.clients.append((sock, addr, None))
        return (sock, addr)

    def process_request_thread(self, started, detached, stopped, request, client_address):
        started.set()
        detached.wait()
        socketserver.ThreadingTCPServer.process_request_thread(self, request, client_address)
        self.close_request(request)
        stopped.set()

    def process_request(self, request, client_address):
        """Start a new thread to process the request."""
        started = threading.Event()
        detached = threading.Event()
        stopped = threading.Event()
        t = TestThread(sync_event=stopped, name='{} -> {}'.format(client_address, self.server_address), target=self.process_request_thread, args=(started, detached, stopped, request, client_address))
        self.clients.pop()
        self.clients.append((request, client_address, t))
        t.set_ignored_exceptions(self.ignored_exceptions)
        t.start()
        started.wait()
        t.pending_exception()
        if debug_threads():
            sys.stderr.write('Client thread {} started\n'.format(t.name))
        detached.set()

    def shutdown_client(self, client):
        sock, addr, connection_thread = client
        self.shutdown_socket(sock)
        if connection_thread is not None:
            if debug_threads():
                sys.stderr.write('Client thread %s will be joined\n' % (connection_thread.name,))
            connection_thread.join()

    def set_ignored_exceptions(self, thread, ignored_exceptions):
        TestingTCPServerMixin.set_ignored_exceptions(self, thread, ignored_exceptions)
        for sock, addr, connection_thread in self.clients:
            if connection_thread is not None:
                connection_thread.set_ignored_exceptions(self.ignored_exceptions)

    def _pending_exception(self, thread):
        for sock, addr, connection_thread in self.clients:
            if connection_thread is not None:
                connection_thread.pending_exception()
        TestingTCPServerMixin._pending_exception(self, thread)