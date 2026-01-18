import requests
from requests.adapters import HTTPAdapter
from .._compat import poolmanager

    A HTTPS Adapter for Python Requests that allows the choice of the SSL/TLS
    version negotiated by Requests. This can be used either to enforce the
    choice of high-security TLS versions (where supported), or to work around
    misbehaving servers that fail to correctly negotiate the default TLS
    version being offered.

    Example usage:

        >>> import requests
        >>> import ssl
        >>> from requests_toolbelt import SSLAdapter
        >>> s = requests.Session()
        >>> s.mount('https://', SSLAdapter(ssl.PROTOCOL_TLSv1))

    You can replace the chosen protocol with any that are available in the
    default Python SSL module. All subsequent requests that match the adapter
    prefix will use the chosen SSL version instead of the default.

    This adapter will also attempt to change the SSL/TLS version negotiated by
    Requests when using a proxy. However, this may not always be possible:
    prior to Requests v2.4.0 the adapter did not have access to the proxy setup
    code. In earlier versions of Requests, this adapter will not function
    properly when used with proxies.
    