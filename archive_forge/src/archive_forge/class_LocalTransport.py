import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
class LocalTransport(transport.Transport):
    """This is the transport agent for local filesystem access."""

    def __init__(self, base):
        """Set the base path where files will be stored."""
        if not base.startswith('file://'):
            raise AssertionError('not a file:// url: %r' % base)
        if base[-1] != '/':
            base = base + '/'
        if sys.platform == 'win32' and base == 'file:///':
            base = ''
            self._local_base = ''
            super().__init__(base)
            return
        super().__init__(base)
        self._local_base = urlutils.local_path_from_url(base)
        if self._local_base[-1] != '/':
            self._local_base = self._local_base + '/'

    def clone(self, offset=None):
        """Return a new LocalTransport with root at self.base + offset
        Because the local filesystem does not require a connection,
        we can just return a new object.
        """
        if offset is None:
            return LocalTransport(self.base)
        else:
            abspath = self.abspath(offset)
            if abspath == 'file://':
                abspath = self.base
            return LocalTransport(abspath)

    def _abspath(self, relative_reference):
        """Return a path for use in os calls.

        Several assumptions are made:
         - relative_reference does not contain '..'
         - relative_reference is url escaped.
        """
        if relative_reference in ('.', ''):
            return osutils.split(self._local_base)[0]
        return self._local_base + urlutils.unescape(relative_reference)

    def abspath(self, relpath):
        """Return the full url to the given relative URL."""
        path = osutils.normpath(osutils.pathjoin(self._local_base, urlutils.unescape(relpath)))
        if sys.platform == 'win32' and self._local_base[1:2] == ':' and (path == '/'):
            path = self._local_base[:3]
        return urlutils.local_path_to_url(path)

    def local_abspath(self, relpath):
        """Transform the given relative path URL into the actual path on disk

        This function only exists for the LocalTransport, since it is
        the only one that has direct local access.
        This is mostly for stuff like WorkingTree which needs to know
        the local working directory.  The returned path will always contain
        forward slashes as the path separator, regardless of the platform.

        This function is quite expensive: it calls realpath which resolves
        symlinks.
        """
        absurl = self.abspath(relpath)
        return urlutils.local_path_from_url(absurl)

    def relpath(self, abspath):
        """Return the local path portion from a given absolute path.
        """
        if abspath is None:
            abspath = '.'
        return urlutils.file_relpath(self.base, abspath)

    def has(self, relpath):
        return os.access(self._abspath(relpath), os.F_OK)

    def get(self, relpath):
        """Get the file at the given relative path.

        :param relpath: The relative path to the file
        """
        canonical_url = self.abspath(relpath)
        if canonical_url in transport._file_streams:
            transport._file_streams[canonical_url].flush()
        try:
            path = self._abspath(relpath)
            return open(path, 'rb')
        except OSError as e:
            if e.errno == errno.EISDIR:
                return transport.LateReadError(relpath)
            self._translate_error(e, path)

    def put_file(self, relpath, f, mode=None):
        """Copy the file-like object into the location.

        :param relpath: Location to put the contents, relative to base.
        :param f:       File-like object.
        :param mode: The mode for the newly created file,
                     None means just use the default
        """
        from ..atomicfile import AtomicFile
        path = relpath
        try:
            path = self._abspath(relpath)
            osutils.check_legal_path(path)
            fp = AtomicFile(path, 'wb', new_mode=mode)
        except OSError as e:
            self._translate_error(e, path)
        try:
            length = self._pump(f, fp)
            fp.commit()
        finally:
            fp.close()
        return length

    def put_bytes(self, relpath: str, raw_bytes: bytes, mode=None):
        """Copy the string into the location.

        :param relpath: Location to put the contents, relative to base.
        :param raw_bytes:   String
        """
        from ..atomicfile import AtomicFile
        if not isinstance(raw_bytes, bytes):
            raise TypeError('raw_bytes must be bytes, not %s' % type(raw_bytes))
        path = relpath
        try:
            path = self._abspath(relpath)
            osutils.check_legal_path(path)
            fp = AtomicFile(path, 'wb', new_mode=mode)
        except OSError as e:
            self._translate_error(e, path)
        try:
            if raw_bytes:
                fp.write(raw_bytes)
            fp.commit()
        finally:
            fp.close()

    def _put_non_atomic_helper(self, relpath, writer, mode=None, create_parent_dir=False, dir_mode=None):
        """Common functionality information for the put_*_non_atomic.

        This tracks all the create_parent_dir stuff.

        :param relpath: the path we are putting to.
        :param writer: A function that takes an os level file descriptor
            and writes whatever data it needs to write there.
        :param mode: The final file mode.
        :param create_parent_dir: Should we be creating the parent directory
            if it doesn't exist?
        """
        abspath = self._abspath(relpath)
        if mode is None:
            local_mode = 438
        else:
            local_mode = mode
        try:
            fd = os.open(abspath, _put_non_atomic_flags, local_mode)
        except OSError as e:
            if not create_parent_dir or e.errno not in (errno.ENOENT, errno.ENOTDIR):
                self._translate_error(e, relpath)
            parent_dir = os.path.dirname(abspath)
            if not parent_dir:
                self._translate_error(e, relpath)
            self._mkdir(parent_dir, mode=dir_mode)
            try:
                fd = os.open(abspath, _put_non_atomic_flags, local_mode)
            except OSError as e:
                self._translate_error(e, relpath)
        try:
            st = os.fstat(fd)
            if mode is not None and mode != S_IMODE(st.st_mode):
                osutils.chmod_if_possible(abspath, mode)
            writer(fd)
        finally:
            os.close(fd)

    def put_file_non_atomic(self, relpath, f, mode=None, create_parent_dir=False, dir_mode=None):
        """Copy the file-like object into the target location.

        This function is not strictly safe to use. It is only meant to
        be used when you already know that the target does not exist.
        It is not safe, because it will open and truncate the remote
        file. So there may be a time when the file has invalid contents.

        :param relpath: The remote location to put the contents.
        :param f:       File-like object.
        :param mode:    Possible access permissions for new file.
                        None means do not set remote permissions.
        :param create_parent_dir: If we cannot create the target file because
                        the parent directory does not exist, go ahead and
                        create it, and then try again.
        """

        def writer(fd):
            self._pump_to_fd(f, fd)
        self._put_non_atomic_helper(relpath, writer, mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)

    def put_bytes_non_atomic(self, relpath: str, raw_bytes: bytes, mode=None, create_parent_dir=False, dir_mode=None):

        def writer(fd):
            if raw_bytes:
                os.write(fd, raw_bytes)
        self._put_non_atomic_helper(relpath, writer, mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)

    def iter_files_recursive(self):
        """Iter the relative paths of files in the transports sub-tree."""
        queue = list(self.list_dir('.'))
        while queue:
            relpath = queue.pop(0)
            st = self.stat(relpath)
            if S_ISDIR(st[ST_MODE]):
                for i, basename in enumerate(self.list_dir(relpath)):
                    queue.insert(i, relpath + '/' + basename)
            else:
                yield relpath

    def _mkdir(self, abspath, mode=None):
        """Create a real directory, filtering through mode"""
        if mode is None:
            local_mode = 511
        else:
            local_mode = mode
        try:
            os.mkdir(abspath, local_mode)
        except OSError as e:
            self._translate_error(e, abspath)
        if mode is not None:
            try:
                osutils.chmod_if_possible(abspath, mode)
            except OSError as e:
                self._translate_error(e, abspath)

    def mkdir(self, relpath, mode=None):
        """Create a directory at the given path."""
        self._mkdir(self._abspath(relpath), mode=mode)

    def open_write_stream(self, relpath, mode=None):
        """See Transport.open_write_stream."""
        abspath = self._abspath(relpath)
        try:
            handle = open(abspath, 'wb')
        except OSError as e:
            self._translate_error(e, abspath)
        handle.truncate()
        if mode is not None:
            self._check_mode_and_size(abspath, handle.fileno(), mode)
        transport._file_streams[self.abspath(relpath)] = handle
        return transport.FileFileStream(self, relpath, handle)

    def _get_append_file(self, relpath, mode=None):
        """Call os.open() for the given relpath"""
        file_abspath = self._abspath(relpath)
        if mode is None:
            local_mode = 438
        else:
            local_mode = mode
        try:
            return (file_abspath, os.open(file_abspath, _append_flags, local_mode))
        except OSError as e:
            self._translate_error(e, relpath)

    def _check_mode_and_size(self, file_abspath, fd, mode=None):
        """Check the mode of the file, and return the current size"""
        st = os.fstat(fd)
        if mode is not None and mode != S_IMODE(st.st_mode):
            osutils.chmod_if_possible(file_abspath, mode)
        return st.st_size

    def append_file(self, relpath, f, mode=None):
        """Append the text in the file-like object into the final location."""
        file_abspath, fd = self._get_append_file(relpath, mode=mode)
        try:
            result = self._check_mode_and_size(file_abspath, fd, mode=mode)
            self._pump_to_fd(f, fd)
        finally:
            os.close(fd)
        return result

    def append_bytes(self, relpath, bytes, mode=None):
        """Append the text in the string into the final location."""
        file_abspath, fd = self._get_append_file(relpath, mode=mode)
        try:
            result = self._check_mode_and_size(file_abspath, fd, mode=mode)
            if bytes:
                os.write(fd, bytes)
        finally:
            os.close(fd)
        return result

    def _pump_to_fd(self, fromfile, to_fd):
        """Copy contents of one file to another."""
        BUFSIZE = 32768
        while True:
            b = fromfile.read(BUFSIZE)
            if not b:
                break
            os.write(to_fd, b)

    def copy(self, rel_from, rel_to):
        """Copy the item at rel_from to the location at rel_to"""
        path_from = self._abspath(rel_from)
        path_to = self._abspath(rel_to)
        import shutil
        try:
            shutil.copy(path_from, path_to)
        except OSError as e:
            self._translate_error(e, path_from)

    def rename(self, rel_from, rel_to):
        path_from = self._abspath(rel_from)
        path_to = self._abspath(rel_to)
        try:
            os.rename(path_from, path_to)
        except OSError as e:
            self._translate_error(e, path_from)

    def move(self, rel_from, rel_to):
        """Move the item at rel_from to the location at rel_to"""
        path_from = self._abspath(rel_from)
        path_to = self._abspath(rel_to)
        try:
            osutils.rename(path_from, path_to)
        except OSError as e:
            self._translate_error(e, path_from)

    def delete(self, relpath):
        """Delete the item at relpath"""
        path = relpath
        try:
            path = self._abspath(relpath)
            os.remove(path)
        except OSError as e:
            self._translate_error(e, path)

    def external_url(self):
        """See breezy.transport.Transport.external_url."""
        return self.base

    def copy_to(self, relpaths, other, mode=None, pb=None):
        """Copy a set of entries from self into another Transport.

        :param relpaths: A list/generator of entries to be copied.
        """
        if isinstance(other, LocalTransport):
            import shutil
            total = self._get_total(relpaths)
            count = 0
            for path in relpaths:
                self._update_pb(pb, 'copy-to', count, total)
                try:
                    mypath = self._abspath(path)
                    otherpath = other._abspath(path)
                    shutil.copy(mypath, otherpath)
                    if mode is not None:
                        osutils.chmod_if_possible(otherpath, mode)
                except OSError as e:
                    self._translate_error(e, path)
                count += 1
            return count
        else:
            return super().copy_to(relpaths, other, mode=mode, pb=pb)

    def listable(self):
        """See Transport.listable."""
        return True

    def list_dir(self, relpath):
        """Return a list of all files at the given location.
        WARNING: many transports do not support this, so trying avoid using
        it if at all possible.
        """
        path = self._abspath(relpath)
        try:
            entries = os.listdir(path)
        except OSError as e:
            self._translate_error(e, path)
        return [urlutils.escape(entry) for entry in entries]

    def stat(self, relpath):
        """Return the stat information for a file.
        """
        path = relpath
        try:
            path = self._abspath(relpath)
            return os.lstat(path)
        except OSError as e:
            self._translate_error(e, path)

    def lock_read(self, relpath):
        """Lock the given file for shared (read) access.
        :return: A lock object, which should be passed to Transport.unlock()
        """
        from breezy.lock import ReadLock
        path = relpath
        try:
            path = self._abspath(relpath)
            return ReadLock(path)
        except OSError as e:
            self._translate_error(e, path)

    def lock_write(self, relpath):
        """Lock the given file for exclusive (write) access.
        WARNING: many transports do not support this, so trying avoid using it

        :return: A lock object, which should be passed to Transport.unlock()
        """
        from breezy.lock import WriteLock
        return WriteLock(self._abspath(relpath))

    def rmdir(self, relpath):
        """See Transport.rmdir."""
        path = relpath
        try:
            path = self._abspath(relpath)
            os.rmdir(path)
        except OSError as e:
            self._translate_error(e, path)

    def symlink(self, source, link_name):
        """See Transport.symlink."""
        abs_link_dirpath = urlutils.dirname(self.abspath(link_name))
        source_rel = urlutils.file_relpath(abs_link_dirpath, self.abspath(source))
        try:
            os.symlink(source_rel, self._abspath(link_name))
        except OSError as e:
            self._translate_error(e, source_rel)

    def readlink(self, relpath):
        """See Transport.readlink."""
        try:
            return osutils.readlink(self._abspath(relpath))
        except OSError as e:
            self._translate_error(e, relpath)
    if osutils.hardlinks_good():

        def hardlink(self, source, link_name):
            """See Transport.link."""
            try:
                os.link(self._abspath(source), self._abspath(link_name))
            except OSError as e:
                self._translate_error(e, source)

    def _can_roundtrip_unix_modebits(self):
        if sys.platform == 'win32':
            return False
        else:
            return True