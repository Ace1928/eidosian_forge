from grpc._cython import cygrpc as _cygrpc
class SSLSessionCache(object):
    """An encapsulation of a session cache used for TLS session resumption.

    Instances of this class can be passed to a Channel as values for the
    grpc.ssl_session_cache option
    """

    def __init__(self, cache):
        self._cache = cache

    def __int__(self):
        return int(self._cache)