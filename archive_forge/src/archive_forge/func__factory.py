from . import pathfilter, register_transport
def _factory(self, url):
    return ChrootTransport(self, url)