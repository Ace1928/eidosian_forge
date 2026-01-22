import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class FakeClient(_SmartClient):
    """Lookalike for _SmartClient allowing testing."""

    def __init__(self, fake_medium_base='fake base'):
        """Create a FakeClient."""
        self.responses = []
        self._calls = []
        self.expecting_body = False
        self._expected_calls = None
        _SmartClient.__init__(self, FakeMedium(self._calls, fake_medium_base))

    def add_expected_call(self, call_name, call_args, response_type, response_args, response_body=None):
        if self._expected_calls is None:
            self._expected_calls = []
        self._expected_calls.append((call_name, call_args))
        self.responses.append((response_type, response_args, response_body))

    def add_success_response(self, *args):
        self.responses.append((b'success', args, None))

    def add_success_response_with_body(self, body, *args):
        self.responses.append((b'success', args, body))
        if self._expected_calls is not None:
            self._expected_calls.append(None)

    def add_error_response(self, *args):
        self.responses.append((b'error', args))

    def add_unknown_method_response(self, verb):
        self.responses.append((b'unknown', verb))

    def finished_test(self):
        if self._expected_calls:
            raise AssertionError('%r finished but was still expecting %r' % (self, self._expected_calls[0]))

    def _get_next_response(self):
        try:
            response_tuple = self.responses.pop(0)
        except IndexError:
            raise AssertionError("{!r} didn't expect any more calls".format(self))
        if response_tuple[0] == b'unknown':
            raise errors.UnknownSmartMethod(response_tuple[1])
        elif response_tuple[0] == b'error':
            raise errors.ErrorFromSmartServer(response_tuple[1])
        return response_tuple

    def _check_call(self, method, args):
        if self._expected_calls is None:
            return
        try:
            next_call = self._expected_calls.pop(0)
        except IndexError:
            raise AssertionError("%r didn't expect any more calls but got %r%r" % (self, method, args))
        if next_call is None:
            return
        if method != next_call[0] or args != next_call[1]:
            raise AssertionError('%r expected %r%r but got %r%r' % (self, next_call[0], next_call[1], method, args))

    def call(self, method, *args):
        self._check_call(method, args)
        self._calls.append(('call', method, args))
        return self._get_next_response()[1]

    def call_expecting_body(self, method, *args):
        self._check_call(method, args)
        self._calls.append(('call_expecting_body', method, args))
        result = self._get_next_response()
        self.expecting_body = True
        return (result[1], FakeProtocol(result[2], self))

    def call_with_body_bytes(self, method, args, body):
        self._check_call(method, args)
        self._calls.append(('call_with_body_bytes', method, args, body))
        result = self._get_next_response()
        return (result[1], FakeProtocol(result[2], self))

    def call_with_body_bytes_expecting_body(self, method, args, body):
        self._check_call(method, args)
        self._calls.append(('call_with_body_bytes_expecting_body', method, args, body))
        result = self._get_next_response()
        self.expecting_body = True
        return (result[1], FakeProtocol(result[2], self))

    def call_with_body_stream(self, args, stream):
        stream = list(stream)
        self._check_call(args[0], args[1:])
        self._calls.append(('call_with_body_stream', args[0], args[1:], stream))
        result = self._get_next_response()
        response_handler = None
        return (result[1], response_handler)