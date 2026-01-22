from __future__ import absolute_import
from socket import error as SocketError
from socket import timeout as SocketTimeout
from ..connection import HTTPConnection, HTTPSConnection
from ..connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import ConnectTimeoutError, NewConnectionError
from ..poolmanager import PoolManager
from ..util.url import parse_url
class SOCKSConnection(HTTPConnection):
    """
    A plain-text HTTP connection that connects via a SOCKS proxy.
    """

    def __init__(self, *args, **kwargs):
        self._socks_options = kwargs.pop('_socks_options')
        super(SOCKSConnection, self).__init__(*args, **kwargs)

    def _new_conn(self):
        """
        Establish a new connection via the SOCKS proxy.
        """
        extra_kw = {}
        if self.source_address:
            extra_kw['source_address'] = self.source_address
        if self.socket_options:
            extra_kw['socket_options'] = self.socket_options
        try:
            conn = socks.create_connection((self.host, self.port), proxy_type=self._socks_options['socks_version'], proxy_addr=self._socks_options['proxy_host'], proxy_port=self._socks_options['proxy_port'], proxy_username=self._socks_options['username'], proxy_password=self._socks_options['password'], proxy_rdns=self._socks_options['rdns'], timeout=self.timeout, **extra_kw)
        except SocketTimeout:
            raise ConnectTimeoutError(self, 'Connection to %s timed out. (connect timeout=%s)' % (self.host, self.timeout))
        except socks.ProxyError as e:
            if e.socket_err:
                error = e.socket_err
                if isinstance(error, SocketTimeout):
                    raise ConnectTimeoutError(self, 'Connection to %s timed out. (connect timeout=%s)' % (self.host, self.timeout))
                else:
                    raise NewConnectionError(self, 'Failed to establish a new connection: %s' % error)
            else:
                raise NewConnectionError(self, 'Failed to establish a new connection: %s' % e)
        except SocketError as e:
            raise NewConnectionError(self, 'Failed to establish a new connection: %s' % e)
        return conn