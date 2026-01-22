from .. import urlutils
from . import Server, Transport, register_transport, unregister_transport
class PathFilteringTransport(Transport):
    """A PathFilteringTransport.

    Please see PathFilteringServer for details.
    """

    def __init__(self, server, base):
        self.server = server
        if not base.endswith('/'):
            base += '/'
        Transport.__init__(self, base)
        self.base_path = self.base[len(self.server.scheme) - 1:]
        self.scheme = self.server.scheme

    def _relpath_from_server_root(self, relpath):
        unfiltered_path = urlutils.URL._combine_paths(self.base_path, relpath)
        if not unfiltered_path.startswith('/'):
            raise ValueError(unfiltered_path)
        return unfiltered_path[1:]

    def _filter(self, relpath):
        return self.server.filter_func(self._relpath_from_server_root(relpath))

    def _call(self, methodname, relpath, *args):
        """Helper for Transport methods of the form:
            operation(path, [other args ...])
        """
        backing_method = getattr(self.server.backing_transport, methodname)
        return backing_method(self._filter(relpath), *args)

    def abspath(self, relpath):
        return self.scheme + self._relpath_from_server_root(relpath)

    def append_file(self, relpath, f, mode=None):
        return self._call('append_file', relpath, f, mode)

    def _can_roundtrip_unix_modebits(self):
        return self.server.backing_transport._can_roundtrip_unix_modebits()

    def clone(self, relpath):
        return self.__class__(self.server, self.abspath(relpath))

    def delete(self, relpath):
        return self._call('delete', relpath)

    def delete_tree(self, relpath):
        return self._call('delete_tree', relpath)

    def external_url(self):
        """See breezy.transport.Transport.external_url."""
        return self.server.backing_transport.external_url()

    def get(self, relpath):
        return self._call('get', relpath)

    def has(self, relpath):
        return self._call('has', relpath)

    def is_readonly(self):
        return self.server.backing_transport.is_readonly()

    def iter_files_recursive(self):
        backing_transport = self.server.backing_transport.clone(self._filter('.'))
        return backing_transport.iter_files_recursive()

    def listable(self):
        return self.server.backing_transport.listable()

    def list_dir(self, relpath):
        return self._call('list_dir', relpath)

    def lock_read(self, relpath):
        return self._call('lock_read', relpath)

    def lock_write(self, relpath):
        return self._call('lock_write', relpath)

    def mkdir(self, relpath, mode=None):
        return self._call('mkdir', relpath, mode)

    def open_write_stream(self, relpath, mode=None):
        return self._call('open_write_stream', relpath, mode)

    def put_file(self, relpath, f, mode=None):
        return self._call('put_file', relpath, f, mode)

    def rename(self, rel_from, rel_to):
        return self._call('rename', rel_from, self._filter(rel_to))

    def rmdir(self, relpath):
        return self._call('rmdir', relpath)

    def stat(self, relpath):
        return self._call('stat', relpath)