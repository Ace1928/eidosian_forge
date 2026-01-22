from . import pathfilter, register_transport
class ChrootTransport(pathfilter.PathFilteringTransport):
    """A ChrootTransport.

    Please see ChrootServer for details.
    """

    def _filter(self, relpath):
        return self._relpath_from_server_root(relpath)