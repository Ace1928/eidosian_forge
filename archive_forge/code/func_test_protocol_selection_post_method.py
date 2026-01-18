import unittest
from wsme import WSRoot
import wsme.protocol
import wsme.rest.protocol
from wsme.root import default_prepare_response_body
from webob import Request
def test_protocol_selection_post_method(self):

    class P(wsme.protocol.Protocol):
        name = 'test'

        def accept(self, r):
            return True
    root = WSRoot()
    root.addprotocol(wsme.rest.protocol.RestProtocol())
    root.addprotocol(P())
    req = Request.blank('/test?check=a&check=b&name=Bob')
    req.headers['Content-Type'] = 'test/fake'
    req.method = 'POST'
    p = root._select_protocol(req)
    assert p.name == 'test'