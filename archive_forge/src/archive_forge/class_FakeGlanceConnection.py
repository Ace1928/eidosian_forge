import os
import routes
import webob
from glance.api.middleware import context
from glance.api.v2 import router
import glance.common.client
class FakeGlanceConnection(object):

    def __init__(self, *args, **kwargs):
        self.sock = FakeSocket()
        self.stub_force_sendfile = kwargs.get('stub_force_sendfile', SENDFILE_SUPPORTED)

    def connect(self):
        return True

    def close(self):
        return True

    def putrequest(self, method, url):
        self.req = webob.Request.blank(url)
        if self.stub_force_sendfile:
            fake_sendfile = FakeSendFile(self.req)
            stubs.Set(sendfile, 'sendfile', fake_sendfile.sendfile)
        self.req.method = method

    def putheader(self, key, value):
        self.req.headers[key] = value

    def endheaders(self):
        hl = [i.lower() for i in self.req.headers.keys()]
        assert not ('content-length' in hl and 'transfer-encoding' in hl), 'Content-Length and Transfer-Encoding are mutually exclusive'

    def send(self, data):
        self.req.body += data.split('\r\n')[1]

    def request(self, method, url, body=None, headers=None):
        self.req = webob.Request.blank(url)
        self.req.method = method
        if headers:
            self.req.headers = headers
        if body:
            self.req.body = body

    def getresponse(self):
        mapper = routes.Mapper()
        api = context.UnauthenticatedContextMiddleware(router.API(mapper))
        res = self.req.get_response(api)

        def fake_reader():
            return res.body
        setattr(res, 'read', fake_reader)
        return res