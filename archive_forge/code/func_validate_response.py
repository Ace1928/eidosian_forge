import platform
import select
import socket
import ssl
import sys
import mock
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3.util import ssl_
from urllib3.util.ssltransport import SSLTransport
def validate_response(provided_response, binary=True):
    assert provided_response is not None
    expected_response = sample_response(binary)
    assert provided_response == expected_response