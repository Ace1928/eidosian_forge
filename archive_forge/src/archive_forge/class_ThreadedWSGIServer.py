import logging
import socket
import socketserver
import sys
from collections import deque
from wsgiref import simple_server
from django.core.exceptions import ImproperlyConfigured
from django.core.handlers.wsgi import LimitedStream
from django.core.wsgi import get_wsgi_application
from django.db import connections
from django.utils.module_loading import import_string
class ThreadedWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    """A threaded version of the WSGIServer"""
    daemon_threads = True

    def __init__(self, *args, connections_override=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.connections_override = connections_override

    def process_request_thread(self, request, client_address):
        if self.connections_override:
            for alias, conn in self.connections_override.items():
                connections[alias] = conn
        super().process_request_thread(request, client_address)

    def _close_connections(self):
        connections.close_all()

    def close_request(self, request):
        self._close_connections()
        super().close_request(request)