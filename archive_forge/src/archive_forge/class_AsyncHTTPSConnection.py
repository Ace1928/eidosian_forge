from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
class AsyncHTTPSConnection:
    """Async HTTP Connection."""
    Request = Request
    Response = AsyncHTTPResponse
    method = 'GET'
    path = '/'
    body = None
    default_ports = {'http': 80, 'https': 443}

    def __init__(self, strict=None, timeout=20.0, http_client=None):
        self.headers = []
        self.timeout = timeout
        self.strict = strict
        self.http_client = http_client or get_client()

    def request(self, method, path, body=None, headers=None):
        self.path = path
        self.method = method
        if body is not None:
            try:
                read = body.read
            except AttributeError:
                self.body = body
            else:
                self.body = read()
        if headers is not None:
            self.headers.extend(list(headers.items()))

    def getrequest(self):
        headers = Headers(self.headers)
        return self.Request(self.path, method=self.method, headers=headers, body=self.body, connect_timeout=self.timeout, request_timeout=self.timeout, validate_cert=False)

    def getresponse(self, callback=None):
        request = self.getrequest()
        request.then(transform(self.Response, callback))
        return self.http_client.add_request(request)

    def set_debuglevel(self, level):
        pass

    def connect(self):
        pass

    def close(self):
        pass

    def putrequest(self, method, path):
        self.method = method
        self.path = path

    def putheader(self, header, value):
        self.headers.append((header, value))

    def endheaders(self):
        pass

    def send(self, data):
        if self.body:
            self.body += data
        else:
            self.body = data

    def __repr__(self):
        return f'<AsyncHTTPConnection: {self.getrequest()!r}>'