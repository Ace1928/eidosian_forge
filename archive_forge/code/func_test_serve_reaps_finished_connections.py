import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def test_serve_reaps_finished_connections(self):
    server, server_thread = self.make_server()
    client_sock1 = self.connect_to_server(server)
    self.say_hello(client_sock1)
    server_handler1, server_side_thread1 = server._active_connections[0]
    client_sock1.close()
    server_side_thread1.join()
    client_sock2 = self.connect_to_server(server)
    self.say_hello(client_sock2)
    server_handler2, server_side_thread2 = server._active_connections[-1]
    client_sock3 = self.connect_to_server(server)
    self.say_hello(client_sock3)
    conns = list(server._active_connections)
    self.assertEqual(2, len(conns))
    self.assertNotEqual((server_handler1, server_side_thread1), conns[0])
    self.assertEqual((server_handler2, server_side_thread2), conns[0])
    client_sock2.close()
    client_sock3.close()
    self.shutdown_server_cleanly(server, server_thread)