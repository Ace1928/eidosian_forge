from collections import abc
import errno
import socket
import ssl
import warnings
import httplib2
import six.moves.http_client
import urllib3
def _patch_add_certificate(pool_manager):
    """Monkey-patches PoolManager to make it accept client certificates."""

    def add_certificate(key, cert, password):
        pool_manager._client_key = key
        pool_manager._client_cert = cert
        pool_manager._client_key_password = password

    def connection_from_host(host, port=None, scheme='http', pool_kwargs=None):
        pool = pool_manager._connection_from_host(host, port, scheme, pool_kwargs)
        pool.key_file = pool_manager._client_key
        pool.cert_file = pool_manager._client_cert
        pool.key_password = pool_manager._client_key_password
        return pool
    pool_manager.add_certificate = add_certificate
    pool_manager.add_certificate(None, None, None)
    pool_manager._connection_from_host = pool_manager.connection_from_host
    pool_manager.connection_from_host = connection_from_host