import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class HttpDavTransport(urllib.HttpTransport):
    """An transport able to put files using http[s] on a DAV server.

    We don't try to implement the whole WebDAV protocol. Just the minimum
    needed for bzr.
    """
    _debuglevel = 0
    _opener_class = DavOpener

    def is_readonly(self):
        """See Transport.is_readonly."""
        return False

    def _raise_http_error(self, url, response, info=None):
        if info is None:
            msg = ''
        else:
            msg = ': ' + info
        raise errors.InvalidHttpResponse(url, 'Unable to handle http code %d%s' % (response.status, msg))

    def open_write_stream(self, relpath, mode=None):
        """See Transport.open_write_stream."""
        self.put_bytes(relpath, b'', mode)
        result = transport.AppendBasedFileStream(self, relpath)
        transport._file_streams[self.abspath(relpath)] = result
        return result

    def put_file(self, relpath, f, mode=None):
        """See Transport.put_file"""
        bytes = f.read()
        self.put_bytes(relpath, bytes, mode=None)
        return len(bytes)

    def put_bytes(self, relpath, bytes, mode=None):
        """Copy the bytes object into the location.

        Tests revealed that contrary to what is said in
        http://www.rfc.net/rfc2068.html, the put is not
        atomic. When putting a file, if the client died, a
        partial file may still exists on the server.

        So we first put a temp file and then move it.

        :param relpath: Location to put the contents, relative to base.
        :param f:       File-like object.
        :param mode:    Not supported by DAV.
        """
        abspath = self._remote_path(relpath)
        stamp = '.tmp.%.9f.%d.%d' % (time.time(), os.getpid(), random.randint(0, 2147483647))
        tmp_relpath = relpath + stamp
        self.put_bytes_non_atomic(tmp_relpath, bytes)
        try:
            self.move(tmp_relpath, relpath)
        except Exception as e:
            exc_type, exc_val, exc_tb = sys.exc_info()
            try:
                self.delete(tmp_relpath)
            except:
                raise exc_type(exc_val).with_traceback(exc_tb)
            raise

    def put_file_non_atomic(self, relpath, f, mode=None, create_parent_dir=False, dir_mode=False):
        self.put_bytes_non_atomic(relpath, f.read(), mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)

    def put_bytes_non_atomic(self, relpath, bytes: bytes, mode=None, create_parent_dir=False, dir_mode=False):
        """See Transport.put_file_non_atomic"""
        abspath = self._remote_path(relpath)
        headers = {'Accept': '*/*', 'Content-type': 'application/octet-stream'}

        def bare_put_file_non_atomic():
            response = self.request('PUT', abspath, body=bytes, headers=headers)
            code = response.status
            if code in (403, 404, 409):
                raise transport.NoSuchFile(abspath)
            elif code not in (200, 201, 204):
                raise self._raise_http_error(abspath, response, 'put file failed')
        try:
            bare_put_file_non_atomic()
        except transport.NoSuchFile:
            if not create_parent_dir:
                raise
            parent_dir = osutils.dirname(relpath)
            if parent_dir:
                self.mkdir(parent_dir, mode=dir_mode)
                return bare_put_file_non_atomic()
            else:
                raise

    def _put_bytes_ranged(self, relpath, bytes, at):
        """Append the file-like object part to the end of the location.

        :param relpath: Location to put the contents, relative to base.
        :param bytes:   A string of bytes to upload
        :param at:      The position in the file to add the bytes
        """
        abspath = self._remote_path(relpath)
        headers = {'Accept': '*/*', 'Content-type': 'application/octet-stream', 'Content-Range': 'bytes %d-%d/*' % (at, at + len(bytes) - 1)}
        response = self.request('PUT', abspath, body=bytes, headers=headers)
        code = response.status
        if code in (403, 404, 409):
            raise transport.NoSuchFile(abspath)
        if code not in (200, 201, 204):
            raise self._raise_http_error(abspath, response, 'put file failed')

    def mkdir(self, relpath, mode=None):
        """See Transport.mkdir"""
        abspath = self._remote_path(relpath)
        response = self.request('MKCOL', abspath)
        code = response.status
        if code == 403:
            raise self._raise_http_error(abspath, response, 'mkdir failed')
        elif code == 405:
            raise transport.FileExists(abspath)
        elif code in (404, 409):
            raise transport.NoSuchFile(abspath)
        elif code != 201:
            raise self._raise_http_error(abspath, response, 'mkdir failed')

    def rename(self, rel_from, rel_to):
        """Rename without special overwriting"""
        abs_from = self._remote_path(rel_from)
        abs_to = self._remote_path(rel_to)
        response = self.request('MOVE', abs_from, headers={'Destination': abs_to, 'Overwrite': 'F'})
        code = response.status
        if code == 404:
            raise transport.NoSuchFile(abs_from)
        if code == 412:
            raise transport.FileExists(abs_to)
        if code == 409:
            raise transport.NoSuchFile(abs_to)
        if code != 201:
            self._raise_http_error(abs_from, response, 'unable to rename to %r' % abs_to)

    def move(self, rel_from, rel_to):
        """See Transport.move"""
        abs_from = self._remote_path(rel_from)
        abs_to = self._remote_path(rel_to)
        response = self.request('MOVE', abs_from, headers={'Destination': abs_to, 'Overwrite': 'T'})
        code = response.status
        if code == 404:
            raise transport.NoSuchFile(abs_from)
        if code == 409:
            raise errors.DirectoryNotEmpty(abs_to)
        if code not in (201, 204):
            self._raise_http_error(abs_from, response, 'unable to move to %r' % abs_to)

    def delete(self, rel_path):
        """
        Delete the item at relpath.

        Note that when a non-empty dir requires to be deleted, a conforming DAV
        server will delete the dir and all its content. That does not normally
        happen in bzr.
        """
        abs_path = self._remote_path(rel_path)
        response = self.request('DELETE', abs_path)
        code = response.status
        if code == 404:
            raise transport.NoSuchFile(abs_path)
        if code not in (200, 204):
            self._raise_http_error(abs_path, response, 'unable to delete')

    def copy(self, rel_from, rel_to):
        """See Transport.copy"""
        abs_from = self._remote_path(rel_from)
        abs_to = self._remote_path(rel_to)
        response = self.request('COPY', abs_from, headers={'Destination': abs_to})
        code = response.status
        if code in (404, 409):
            raise transport.NoSuchFile(abs_from)
        if code not in (201, 204):
            self._raise_http_error(abs_from, response, 'unable to copy from %r to %r' % (abs_from, abs_to))

    def copy_to(self, relpaths, other, mode=None, pb=None):
        """Copy a set of entries from self into another Transport.

        :param relpaths: A list/generator of entries to be copied.
        """
        return transport.Transport.copy_to(self, relpaths, other, mode=mode, pb=pb)

    def listable(self):
        """See Transport.listable."""
        return True

    def list_dir(self, relpath):
        """
        Return a list of all files at the given location.
        """
        return [elt[0] for elt in self._list_tree(relpath, 1)]

    def _list_tree(self, relpath, depth):
        abspath = self._remote_path(relpath)
        propfind = b'<?xml version="1.0" encoding="utf-8" ?>\n   <D:propfind xmlns:D="DAV:">\n     <D:allprop/>\n   </D:propfind>\n'
        response = self.request('PROPFIND', abspath, body=propfind, headers={'Depth': '{}'.format(depth), 'Content-Type': 'application/xml; charset="utf-8"'})
        code = response.status
        if code == 404:
            raise transport.NoSuchFile(abspath)
        if code == 409:
            raise transport.NoSuchFile(abspath)
        if code != 207:
            self._raise_http_error(abspath, response, 'unable to list  %r directory' % abspath)
        return _extract_dir_content(abspath, response)

    def lock_write(self, relpath):
        """Lock the given file for exclusive access.
        :return: A lock object, which should be passed to Transport.unlock()
        """
        return self.lock_read(relpath)

    def rmdir(self, relpath):
        """See Transport.rmdir."""
        content = self.list_dir(relpath)
        if len(content) > 0:
            raise errors.DirectoryNotEmpty(self._remote_path(relpath))
        self.delete(relpath)

    def stat(self, relpath):
        """See Transport.stat.

        We provide a limited implementation for bzr needs.
        """
        abspath = self._remote_path(relpath)
        propfind = b'<?xml version="1.0" encoding="utf-8" ?>\n   <D:propfind xmlns:D="DAV:">\n     <D:allprop/>\n   </D:propfind>\n'
        response = self.request('PROPFIND', abspath, body=propfind, headers={'Depth': '0', 'Content-Type': 'application/xml; charset="utf-8"'})
        code = response.status
        if code == 404:
            raise transport.NoSuchFile(abspath)
        if code == 409:
            raise transport.NoSuchFile(abspath)
        if code != 207:
            self._raise_http_error(abspath, response, 'unable to list  %r directory' % abspath)
        return _extract_stat_info(abspath, response)

    def iter_files_recursive(self):
        """Walk the relative paths of all files in this transport."""
        tree = self._list_tree('.', 'Infinity')
        for name, is_dir, size, is_exex in tree:
            if not is_dir:
                yield name

    def append_file(self, relpath, f, mode=None):
        """See Transport.append_file"""
        return self.append_bytes(relpath, f.read(), mode=mode)

    def append_bytes(self, relpath, bytes, mode=None):
        """See Transport.append_bytes"""
        if self._range_hint is not None:
            before = self._append_by_head_put(relpath, bytes)
        else:
            before = self._append_by_get_put(relpath, bytes)
        return before

    def _append_by_head_put(self, relpath, bytes):
        """Append without getting the whole file.

        When the server allows it, a 'Content-Range' header can be specified.
        """
        response = self._head(relpath)
        code = response.status
        if code == 404:
            relpath_size = 0
        else:
            relpath_size = int(response.getheader('Content-Length', 0))
            if relpath_size == 0:
                trace.mutter('if %s is not empty, the server is buggy' % relpath)
        if relpath_size:
            self._put_bytes_ranged(relpath, bytes, relpath_size)
        else:
            self.put_bytes(relpath, bytes)
        return relpath_size

    def _append_by_get_put(self, relpath, bytes):
        full_data = StringIO()
        try:
            data = self.get(relpath)
            full_data.write(data.read())
        except transport.NoSuchFile:
            pass
        before = full_data.tell()
        full_data.write(bytes)
        full_data.seek(0)
        self.put_file(relpath, full_data)
        return before

    def get_smart_medium(self):
        raise errors.NoSmartMedium(self)