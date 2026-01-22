import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
def _get_cookies(self, value, domains, max_age, path, secure, httponly, samesite):
    """Internal function

        This returns a list of cookies that are valid HTTP Headers.

        :environ: The request environment
        :value: The value to store in the cookie
        :domains: The domains, overrides any set in the CookieProfile
        :max_age: The max_age, overrides any set in the CookieProfile
        :path: The path, overrides any set in the CookieProfile
        :secure: Set this cookie to secure, overrides any set in CookieProfile
        :httponly: Set this cookie to HttpOnly, overrides any set in CookieProfile
        :samesite: Set this cookie to be for only the same site, overrides any
                   set in CookieProfile.

        """
    if domains is _default:
        domains = self.domains
    if max_age is _default:
        max_age = self.max_age
    if path is _default:
        path = self.path
    if secure is _default:
        secure = self.secure
    if httponly is _default:
        httponly = self.httponly
    if samesite is _default:
        samesite = self.samesite
    if value is not None and len(value) > 4093:
        raise ValueError('Cookie value is too long to store (%s bytes)' % len(value))
    cookies = []
    if not domains:
        cookievalue = make_cookie(self.cookie_name, value, path=path, max_age=max_age, httponly=httponly, samesite=samesite, secure=secure)
        cookies.append(('Set-Cookie', cookievalue))
    else:
        for domain in domains:
            cookievalue = make_cookie(self.cookie_name, value, path=path, domain=domain, max_age=max_age, httponly=httponly, samesite=samesite, secure=secure)
            cookies.append(('Set-Cookie', cookievalue))
    return cookies