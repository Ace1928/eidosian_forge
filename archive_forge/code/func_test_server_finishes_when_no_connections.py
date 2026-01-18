import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_server_finishes_when_no_connections(self):
    """the server thread exits even if there are no connections"""
    server = Server.basic_response_server()
    with server:
        pass
    assert len(server.handler_results) == 0