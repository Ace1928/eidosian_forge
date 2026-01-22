from collections import abc
import errno
import socket
import ssl
import warnings
import httplib2
import six.moves.http_client
import urllib3
def _map_exception(e):
    """Maps an exception from urlib3 to httplib2."""
    if isinstance(e, urllib3.exceptions.MaxRetryError):
        if not e.reason:
            return e
        e = e.reason
    message = e.args[0] if e.args else ''
    if isinstance(e, urllib3.exceptions.ResponseError):
        if 'too many redirects' in message:
            return httplib2.RedirectLimit(message)
    if isinstance(e, urllib3.exceptions.NewConnectionError):
        if 'Name or service not known' in message or 'nodename nor servname provided, or not known' in message:
            return httplib2.ServerNotFoundError('Unable to find hostname.')
        if 'Connection refused' in message:
            return socket.error((errno.ECONNREFUSED, 'Connection refused'))
    if isinstance(e, urllib3.exceptions.DecodeError):
        return httplib2.FailedToDecompressContent('Content purported as compressed but not uncompressable.', httplib2.Response({'status': 500}), '')
    if isinstance(e, urllib3.exceptions.TimeoutError):
        return socket.timeout('timed out')
    if isinstance(e, urllib3.exceptions.ConnectTimeoutError):
        return socket.timeout('connect timed out')
    if isinstance(e, urllib3.exceptions.SSLError):
        return ssl.SSLError(*e.args)
    return e