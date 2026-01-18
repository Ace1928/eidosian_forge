import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_server_finishes_on_error(self):
    """the server thread exits even if an exception exits the context manager"""
    server = Server.basic_response_server()
    with pytest.raises(Exception):
        with server:
            raise Exception()
    assert len(server.handler_results) == 0