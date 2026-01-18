import unittest
from wsme import WSRoot
import wsme.protocol
import wsme.rest.protocol
from wsme.root import default_prepare_response_body
from webob import Request
def test_protocol_selection_content_type_mismatch(self):
    """Verify that we get a 415 error on wrong Content-Type header."""

    class P(wsme.protocol.Protocol):
        name = 'test'

        def accept(self, r):
            return False
    root = WSRoot()
    root.addprotocol(wsme.rest.protocol.RestProtocol())
    root.addprotocol(P())
    req = Request.blank('/test?check=a&check=b&name=Bob')
    req.method = 'POST'
    req.headers['Content-Type'] = 'test/unsupported'
    res = root._handle_request(req)
    assert res.status_int == 415
    assert res.content_type == 'text/plain'
    assert res.text.startswith('Unacceptable Content-Type: test/unsupported not in'), req.text