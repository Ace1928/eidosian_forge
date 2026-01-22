import httplib2
class NestedTransport(httplib2.Http):
    """Extends and composes an inner httplib2.Http transport."""

    def __init__(self, source_transport):
        self.source_transport = source_transport

    def __getstate__(self):
        raise NotImplementedError()

    def __setstate__(self, state):
        raise NotImplementedError()

    def add_credentials(self, *args, **kwargs):
        self.source_transport.add_credentials(*args, **kwargs)

    def add_certificate(self, *args, **kwargs):
        self.source_transport.add_certificate(*args, **kwargs)

    def clear_credentials(self):
        self.source_transport.clear_credentials()

    def request(self, *args, **kwargs):
        return self.source_transport.request(*args, **kwargs)