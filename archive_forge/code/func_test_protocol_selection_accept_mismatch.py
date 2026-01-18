import unittest
from wsme import WSRoot
import wsme.protocol
import wsme.rest.protocol
from wsme.root import default_prepare_response_body
from webob import Request
def test_protocol_selection_accept_mismatch(self):
    """Verify that we get a 406 error on wrong Accept header."""

    class P(wsme.protocol.Protocol):
        name = 'test'

        def accept(self, r):
            return False
    root = WSRoot()
    root.addprotocol(wsme.rest.protocol.RestProtocol())
    root.addprotocol(P())
    req = Request.blank('/test?check=a&check=b&name=Bob')
    req.method = 'GET'
    res = root._handle_request(req)
    assert res.status_int == 406
    assert res.content_type == 'text/plain'
    assert res.text.startswith('None of the following protocols can handle this request'), req.text