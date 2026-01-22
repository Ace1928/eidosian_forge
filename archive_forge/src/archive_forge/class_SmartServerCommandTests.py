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
class SmartServerCommandTests(tests.TestCaseWithTransport):
    """Tests that call directly into the command objects, bypassing the network
    and the request dispatching.

    Note: these tests are rudimentary versions of the command object tests in
    test_smart.py.
    """

    def test_hello(self):
        cmd = _mod_request.HelloRequest(None, '/')
        response = cmd.execute()
        self.assertEqual((b'ok', b'2'), response.args)
        self.assertEqual(None, response.body)

    def test_get_bundle(self):
        from breezy.bzr.bundle import serializer
        wt = self.make_branch_and_tree('.')
        self.build_tree_contents([('hello', b'hello world')])
        wt.add('hello')
        rev_id = wt.commit('add hello')
        cmd = _mod_request.GetBundleRequest(self.get_transport(), '/')
        response = cmd.execute(b'.', rev_id)
        bundle = serializer.read_bundle(BytesIO(response.body))
        self.assertEqual((), response.args)