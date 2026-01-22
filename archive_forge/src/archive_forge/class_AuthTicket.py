import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
class AuthTicket(object):
    """
    This class represents an authentication token.  You must pass in
    the shared secret, the userid, and the IP address.  Optionally you
    can include tokens (a list of strings, representing role names),
    'user_data', which is arbitrary data available for your own use in
    later scripts.  Lastly, you can override the timestamp, cookie name,
    whether to secure the cookie and the digest algorithm (for details
    look at ``AuthTKTMiddleware``).

    Once you provide all the arguments, use .cookie_value() to
    generate the appropriate authentication ticket.  .cookie()
    generates a Cookie object, the str() of which is the complete
    cookie header to be sent.

    CGI usage::

        token = auth_tkt.AuthTick('sharedsecret', 'username',
            os.environ['REMOTE_ADDR'], tokens=['admin'])
        print('Status: 200 OK')
        print('Content-type: text/html')
        print(token.cookie())
        print("")
        ... redirect HTML ...

    Webware usage::

        token = auth_tkt.AuthTick('sharedsecret', 'username',
            self.request().environ()['REMOTE_ADDR'], tokens=['admin'])
        self.response().setCookie('auth_tkt', token.cookie_value())

    Be careful not to do an HTTP redirect after login; use meta
    refresh or Javascript -- some browsers have bugs where cookies
    aren't saved when set on a redirect.
    """

    def __init__(self, secret, userid, ip, tokens=(), user_data='', time=None, cookie_name='auth_tkt', secure=False, digest_algo=DEFAULT_DIGEST):
        self.secret = secret
        self.userid = userid
        self.ip = ip
        if not isinstance(tokens, str):
            tokens = ','.join(tokens)
        self.tokens = tokens
        self.user_data = user_data
        if time is None:
            self.time = time_mod.time()
        else:
            self.time = time
        self.cookie_name = cookie_name
        self.secure = secure
        if isinstance(digest_algo, bytes):
            self.digest_algo = getattr(hashlib, digest_algo)
        else:
            self.digest_algo = digest_algo

    def digest(self):
        return calculate_digest(self.ip, self.time, self.secret, self.userid, self.tokens, self.user_data, self.digest_algo)

    def cookie_value(self):
        v = b'%s%08x%s!' % (self.digest(), int(self.time), maybe_encode(url_quote(self.userid)))
        if self.tokens:
            v += maybe_encode(self.tokens) + b'!'
        v += maybe_encode(self.user_data)
        return v

    def cookie(self):
        c = SimpleCookie()
        import base64
        cookie_value = base64.b64encode(self.cookie_value())
        c[self.cookie_name] = cookie_value
        c[self.cookie_name]['path'] = '/'
        if self.secure:
            c[self.cookie_name]['secure'] = 'true'
        return c