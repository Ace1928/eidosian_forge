from collections import abc
import errno
import socket
import ssl
import warnings
import httplib2
import six.moves.http_client
import urllib3
def _default_make_pool(http, proxy_info):
    """Creates a urllib3.PoolManager object that has SSL verification enabled."""
    if not http.ca_certs:
        http.ca_certs = httplib2.CA_CERTS
    ssl_disabled = http.disable_ssl_certificate_validation
    cert_reqs = 'CERT_REQUIRED' if http.ca_certs and (not ssl_disabled) else None
    if isinstance(proxy_info, abc.Callable):
        proxy_info = proxy_info()
    if proxy_info:
        if proxy_info.proxy_user and proxy_info.proxy_pass:
            proxy_url = 'http://{}:{}@{}:{}/'.format(proxy_info.proxy_user, proxy_info.proxy_pass, proxy_info.proxy_host, proxy_info.proxy_port)
            proxy_headers = urllib3.util.request.make_headers(proxy_basic_auth='{}:{}'.format(proxy_info.proxy_user, proxy_info.proxy_pass))
        else:
            proxy_url = 'http://{}:{}/'.format(proxy_info.proxy_host, proxy_info.proxy_port)
            proxy_headers = {}
        return urllib3.ProxyManager(proxy_url=proxy_url, proxy_headers=proxy_headers, ca_certs=http.ca_certs, cert_reqs=cert_reqs)
    return urllib3.PoolManager(ca_certs=http.ca_certs, cert_reqs=cert_reqs)