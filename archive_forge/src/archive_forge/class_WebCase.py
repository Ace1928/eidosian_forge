from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
class WebCase(unittest.TestCase):
    """Helper web test suite base."""
    HOST = '127.0.0.1'
    PORT = 8000
    HTTP_CONN = http.client.HTTPConnection
    PROTOCOL = 'HTTP/1.1'
    scheme = 'http'
    url = None
    ssl_context = None
    status = None
    headers = None
    body = None
    encoding = 'utf-8'
    time = None

    @property
    def _Conn(self):
        """Return HTTPConnection or HTTPSConnection based on self.scheme.

        * from :py:mod:`python:http.client`.
        """
        cls_name = '{scheme}Connection'.format(scheme=self.scheme.upper())
        return getattr(http.client, cls_name)

    def get_conn(self, auto_open=False):
        """Return a connection to our HTTP server."""
        conn = self._Conn(self.interface(), self.PORT)
        conn.auto_open = auto_open
        conn.connect()
        return conn

    def set_persistent(self, on=True, auto_open=False):
        """Make our HTTP_CONN persistent (or not).

        If the 'on' argument is True (the default), then self.HTTP_CONN
        will be set to an instance of HTTP(S)?Connection
        to persist across requests.
        As this class only allows for a single open connection, if
        self already has an open connection, it will be closed.
        """
        try:
            self.HTTP_CONN.close()
        except (TypeError, AttributeError):
            pass
        self.HTTP_CONN = self.get_conn(auto_open=auto_open) if on else self._Conn

    @property
    def persistent(self):
        """Presence of the persistent HTTP connection."""
        return hasattr(self.HTTP_CONN, '__class__')

    @persistent.setter
    def persistent(self, on):
        self.set_persistent(on)

    def interface(self):
        """Return an IP address for a client connection.

        If the server is listening on '0.0.0.0' (INADDR_ANY)
        or '::' (IN6ADDR_ANY), this will return the proper localhost.
        """
        return interface(self.HOST)

    def getPage(self, url, headers=None, method='GET', body=None, protocol=None, raise_subcls=()):
        """Open the url with debugging support.

        Return status, headers, body.

        url should be the identifier passed to the server, typically a
        server-absolute path and query string (sent between method and
        protocol), and should only be an absolute URI if proxy support is
        enabled in the server.

        If the application under test generates absolute URIs, be sure
        to wrap them first with :py:func:`strip_netloc`::

            >>> class MyAppWebCase(WebCase):
            ...     def getPage(url, *args, **kwargs):
            ...         super(MyAppWebCase, self).getPage(
            ...             cheroot.test.webtest.strip_netloc(url),
            ...             *args, **kwargs
            ...         )

        ``raise_subcls`` is passed through to :py:func:`openURL`.
        """
        ServerError.on = False
        if isinstance(url, str):
            url = url.encode('utf-8')
        if isinstance(body, str):
            body = body.encode('utf-8')
        raise_subcls = raise_subcls or ()
        self.url = url
        self.time = None
        start = time.time()
        result = openURL(url, headers, method, body, self.HOST, self.PORT, self.HTTP_CONN, protocol or self.PROTOCOL, raise_subcls=raise_subcls, ssl_context=self.ssl_context)
        self.time = time.time() - start
        self.status, self.headers, self.body = result
        self.cookies = [('Cookie', v) for k, v in self.headers if k.lower() == 'set-cookie']
        if ServerError.on:
            raise ServerError()
        return result

    @NonDataProperty
    def interactive(self):
        """Determine whether tests are run in interactive mode.

        Load interactivity setting from environment, where
        the value can be numeric or a string like true or
        False or 1 or 0.
        """
        env_str = os.environ.get('WEBTEST_INTERACTIVE', 'True')
        is_interactive = bool(json.loads(env_str.lower()))
        if is_interactive:
            warnings.warn('Interactive test failure interceptor support via WEBTEST_INTERACTIVE environment variable is deprecated.', DeprecationWarning)
        return is_interactive
    console_height = 30

    def _handlewebError(self, msg):
        print('')
        print('    ERROR: %s' % msg)
        if not self.interactive:
            raise self.failureException(msg)
        p = '    Show: [B]ody [H]eaders [S]tatus [U]RL; [I]gnore, [R]aise, or sys.e[X]it >> '
        sys.stdout.write(p)
        sys.stdout.flush()
        while True:
            i = getchar().upper()
            if not isinstance(i, type('')):
                i = i.decode('ascii')
            if i not in 'BHSUIRX':
                continue
            print(i.upper())
            if i == 'B':
                for x, line in enumerate(self.body.splitlines()):
                    if (x + 1) % self.console_height == 0:
                        sys.stdout.write('<-- More -->\r')
                        m = getchar().lower()
                        sys.stdout.write('            \r')
                        if m == 'q':
                            break
                    print(line)
            elif i == 'H':
                pprint.pprint(self.headers)
            elif i == 'S':
                print(self.status)
            elif i == 'U':
                print(self.url)
            elif i == 'I':
                return
            elif i == 'R':
                raise self.failureException(msg)
            elif i == 'X':
                sys.exit()
            sys.stdout.write(p)
            sys.stdout.flush()

    @property
    def status_code(self):
        """Integer HTTP status code."""
        return int(self.status[:3])

    def status_matches(self, expected):
        """Check whether actual status matches expected."""
        actual = self.status_code if isinstance(expected, int) else self.status
        return expected == actual

    def assertStatus(self, status, msg=None):
        """Fail if self.status != status.

        status may be integer code, exact string status, or
        iterable of allowed possibilities.
        """
        if any(map(self.status_matches, always_iterable(status))):
            return
        tmpl = 'Status {self.status} does not match {status}'
        msg = msg or tmpl.format(**locals())
        self._handlewebError(msg)

    def assertHeader(self, key, value=None, msg=None):
        """Fail if (key, [value]) not in self.headers."""
        lowkey = key.lower()
        for k, v in self.headers:
            if k.lower() == lowkey:
                if value is None or str(value) == v:
                    return v
        if msg is None:
            if value is None:
                msg = '%r not in headers' % key
            else:
                msg = '%r:%r not in headers' % (key, value)
        self._handlewebError(msg)

    def assertHeaderIn(self, key, values, msg=None):
        """Fail if header indicated by key doesn't have one of the values."""
        lowkey = key.lower()
        for k, v in self.headers:
            if k.lower() == lowkey:
                matches = [value for value in values if str(value) == v]
                if matches:
                    return matches
        if msg is None:
            msg = '%(key)r not in %(values)r' % vars()
        self._handlewebError(msg)

    def assertHeaderItemValue(self, key, value, msg=None):
        """Fail if the header does not contain the specified value."""
        actual_value = self.assertHeader(key, msg=msg)
        header_values = map(str.strip, actual_value.split(','))
        if value in header_values:
            return value
        if msg is None:
            msg = '%r not in %r' % (value, header_values)
        self._handlewebError(msg)

    def assertNoHeader(self, key, msg=None):
        """Fail if key in self.headers."""
        lowkey = key.lower()
        matches = [k for k, v in self.headers if k.lower() == lowkey]
        if matches:
            if msg is None:
                msg = '%r in headers' % key
            self._handlewebError(msg)

    def assertNoHeaderItemValue(self, key, value, msg=None):
        """Fail if the header contains the specified value."""
        lowkey = key.lower()
        hdrs = self.headers
        matches = [k for k, v in hdrs if k.lower() == lowkey and v == value]
        if matches:
            if msg is None:
                msg = '%r:%r in %r' % (key, value, hdrs)
            self._handlewebError(msg)

    def assertBody(self, value, msg=None):
        """Fail if value != self.body."""
        if isinstance(value, str):
            value = value.encode(self.encoding)
        if value != self.body:
            if msg is None:
                msg = 'expected body:\n%r\n\nactual body:\n%r' % (value, self.body)
            self._handlewebError(msg)

    def assertInBody(self, value, msg=None):
        """Fail if value not in self.body."""
        if isinstance(value, str):
            value = value.encode(self.encoding)
        if value not in self.body:
            if msg is None:
                msg = '%r not in body: %s' % (value, self.body)
            self._handlewebError(msg)

    def assertNotInBody(self, value, msg=None):
        """Fail if value in self.body."""
        if isinstance(value, str):
            value = value.encode(self.encoding)
        if value in self.body:
            if msg is None:
                msg = '%r found in body' % value
            self._handlewebError(msg)

    def assertMatchesBody(self, pattern, msg=None, flags=0):
        """Fail if value (a regex pattern) is not in self.body."""
        if isinstance(pattern, str):
            pattern = pattern.encode(self.encoding)
        if re.search(pattern, self.body, flags) is None:
            if msg is None:
                msg = 'No match for %r in body' % pattern
            self._handlewebError(msg)