from . import pathfilter, register_transport
class ChrootServer(pathfilter.PathFilteringServer):
    """User space 'chroot' facility.

    The server's get_url returns the url for a chroot transport mapped to the
    backing transport. The url is of the form chroot-xxx:/// so parent
    directories of the backing transport are not visible. The chroot url will
    not allow '..' sequences to result in requests to the chroot affecting
    directories outside the backing transport.

    PathFilteringServer does all the path sanitation needed to enforce a
    chroot, so this is a simple subclass of PathFilteringServer that ignores
    filter_func.
    """

    def __init__(self, backing_transport):
        pathfilter.PathFilteringServer.__init__(self, backing_transport, None)

    def _factory(self, url):
        return ChrootTransport(self, url)

    def start_server(self):
        self.scheme = 'chroot-%d:///' % id(self)
        register_transport(self.scheme, self._factory)