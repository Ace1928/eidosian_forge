import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError

        Test when we have auth info in url, i.e.
        socks5://user:pass@host:port and no username/password as params
        