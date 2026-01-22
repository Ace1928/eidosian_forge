import cgi
import hashlib
import hmac
from http.cookies import SimpleCookie
import logging
import time
from typing import Optional
from urllib.parse import parse_qs
from urllib.parse import quote
from saml2 import BINDING_HTTP_ARTIFACT
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import BINDING_URI
from saml2 import SAMLError
from saml2 import time_util
class SeeOther(Response):
    _template = '<html>\n<head><title>Redirecting to %s</title></head>\n<body>\nYou are being redirected to <a href="%s">%s</a>\n</body>\n</html>'
    _status = '303 See Other'

    def __call__(self, environ, start_response, **kwargs):
        location = ''
        if self.message:
            location = self.message
            self.headers.append(('location', location))
        else:
            for param, item in self.headers:
                if param == 'location':
                    location = item
                    break
        start_response(self.status, self.headers)
        return self.response((location, location, location))