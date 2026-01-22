import hmac, base64, random, time, warnings
from functools import reduce
from paste.request import get_cookies
class AuthCookieHandler(object):
    """
    the actual handler that should be put in your middleware stack

    This middleware uses cookies to stash-away a previously authenticated
    user (and perhaps other variables) so that re-authentication is not
    needed.  This does not implement sessions; and therefore N servers
    can be syncronized to accept the same saved authentication if they
    all use the same cookie_name and secret.

    By default, this handler scans the `environ` for the REMOTE_USER
    and REMOTE_SESSION key; if found, it is stored. It can be
    configured to scan other `environ` keys as well -- but be careful
    not to exceed 2-3k (so that the encoded and signed cookie does not
    exceed 4k). You can ask it to handle other environment variables
    by doing:

       ``environ['paste.auth.cookie'].append('your.environ.variable')``


    Constructor Arguments:

        ``application``

            This is the wrapped application which will have access to
            the ``environ['REMOTE_USER']`` restored by this middleware.

        ``cookie_name``

            The name of the cookie used to store this content, by default
            it is ``PASTE_AUTH_COOKIE``.

        ``scanlist``

            This is the initial set of ``environ`` keys to
            save/restore to the signed cookie.  By default is consists
            only of ``REMOTE_USER`` and ``REMOTE_SESSION``; any tuple
            or list of environment keys will work.  However, be
            careful, as the total saved size is limited to around 3k.

        ``signer``

            This is the signer object used to create the actual cookie
            values, by default, it is ``AuthCookieSigner`` and is passed
            the remaining arguments to this function: ``secret``,
            ``timeout``, and ``maxlen``.

    At this time, each cookie is individually signed.  To store more
    than the 4k of data; it is possible to sub-class this object to
    provide different ``environ_name`` and ``cookie_name``
    """
    environ_name = 'paste.auth.cookie'
    cookie_name = 'PASTE_AUTH_COOKIE'
    signer_class = AuthCookieSigner
    environ_class = AuthCookieEnviron

    def __init__(self, application, cookie_name=None, scanlist=None, signer=None, secret=None, timeout=None, maxlen=None):
        if not signer:
            signer = self.signer_class(secret, timeout, maxlen)
        self.signer = signer
        self.scanlist = scanlist or ('REMOTE_USER', 'REMOTE_SESSION')
        self.application = application
        self.cookie_name = cookie_name or self.cookie_name

    def __call__(self, environ, start_response):
        if self.environ_name in environ:
            raise AssertionError('AuthCookie already installed!')
        scanlist = self.environ_class(self, self.scanlist)
        jar = get_cookies(environ)
        if self.cookie_name in jar:
            content = self.signer.auth(jar[self.cookie_name].value)
            if content:
                for pair in content.split(';'):
                    k, v = pair.split('=')
                    k = decode(k)
                    if k not in scanlist:
                        scanlist.append(k)
                    if k in environ:
                        continue
                    environ[k] = decode(v)
                    if 'REMOTE_USER' == k:
                        environ['AUTH_TYPE'] = 'cookie'
        environ[self.environ_name] = scanlist
        if 'paste.httpexceptions' in environ:
            warnings.warn('Since paste.httpexceptions is hooked in your processing chain before paste.auth.cookie, if an HTTPRedirection is raised, the cookies this module sets will not be included in your response.\n')

        def response_hook(status, response_headers, exc_info=None):
            """
            Scan the environment for keys specified in the scanlist,
            pack up their values, signs the content and issues a cookie.
            """
            scanlist = environ.get(self.environ_name)
            assert scanlist and isinstance(scanlist, self.environ_class)
            content = []
            for k in scanlist:
                v = environ.get(k)
                if v is not None:
                    if type(v) is not str:
                        raise ValueError('The value of the environmental variable %r is not a str (only str is allowed; got %r)' % (k, v))
                    content.append('%s=%s' % (encode(k), encode(v)))
            if content:
                content = ';'.join(content)
                content = self.signer.sign(content)
                content = content.decode('utf8')
                cookie = '%s=%s; Path=/;' % (self.cookie_name, content)
                if 'https' == environ['wsgi.url_scheme']:
                    cookie += ' secure;'
                response_headers.append(('Set-Cookie', cookie))
            return start_response(status, response_headers, exc_info)
        return self.application(environ, response_hook)