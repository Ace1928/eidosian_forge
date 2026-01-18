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
def validate_request(provided_request, binary=True):
    assert provided_request is not None
    expected_request = sample_request(binary)
    assert provided_request == expected_request