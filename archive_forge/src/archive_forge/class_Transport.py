import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class Transport:
    """This class encapsulates methods for retrieving or putting a file
    from/to a storage location.

    :ivar base: Base URL for the transport; should always end in a slash.
    """
    _max_readv_combine = 50
    _bytes_to_read_before_seek = 0
    hooks = TransportHooks()
    base: str

    def __init__(self, base):
        super().__init__()
        self.base = base
        self._raw_base, self._segment_parameters = urlutils.split_segment_parameters(base)

    def _translate_error(self, e, path, raise_generic=True):
        """Translate an IOError or OSError into an appropriate bzr error.

        This handles things like ENOENT, ENOTDIR, EEXIST, and EACCESS
        """
        if getattr(e, 'errno', None) is not None:
            if e.errno in (errno.ENOENT, errno.ENOTDIR):
                raise NoSuchFile(path, extra=e)
            elif e.errno == errno.EINVAL:
                mutter('EINVAL returned on path {}: {!r}'.format(path, e))
                raise NoSuchFile(path, extra=e)
            if sys.platform == 'win32' and e.errno in (errno.ESRCH, 267):
                raise NoSuchFile(path, extra=e)
            if e.errno == errno.EEXIST:
                raise FileExists(path, extra=e)
            if e.errno == errno.EACCES:
                raise errors.PermissionDenied(path, extra=e)
            if e.errno == errno.ENOTEMPTY:
                raise errors.DirectoryNotEmpty(path, extra=e)
            if e.errno == errno.EBUSY:
                raise errors.ResourceBusy(path, extra=e)
        if raise_generic:
            raise errors.TransportError(orig_error=e)

    def clone(self, offset=None):
        """Return a new Transport object, cloned from the current location,
        using a subdirectory or parent directory. This allows connections
        to be pooled, rather than a new one needed for each subdir.
        """
        raise NotImplementedError(self.clone)

    def create_prefix(self, mode=None):
        """Create all the directories leading down to self.base."""
        cur_transport = self
        needed = [cur_transport]
        while True:
            new_transport = cur_transport.clone('..')
            if new_transport.base == cur_transport.base:
                raise errors.CommandError('Failed to create path prefix for %s.' % cur_transport.base)
            try:
                new_transport.mkdir('.', mode=mode)
            except NoSuchFile:
                needed.append(new_transport)
                cur_transport = new_transport
            except FileExists:
                break
            else:
                break
        while needed:
            cur_transport = needed.pop()
            cur_transport.ensure_base(mode=mode)

    def ensure_base(self, mode=None):
        """Ensure that the directory this transport references exists.

        This will create a directory if it doesn't exist.
        :return: True if the directory was created, False otherwise.
        """
        try:
            self.mkdir('.', mode=mode)
        except (FileExists, errors.PermissionDenied):
            return False
        except errors.TransportNotPossible:
            if self.has('.'):
                return False
            raise
        else:
            return True

    def external_url(self):
        """Return a URL for self that can be given to an external process.

        There is no guarantee that the URL can be accessed from a different
        machine - e.g. file:/// urls are only usable on the local machine,
        sftp:/// urls when the server is only bound to localhost are only
        usable from localhost etc.

        NOTE: This method may remove security wrappers (e.g. on chroot
        transports) and thus should *only* be used when the result will not
        be used to obtain a new transport within breezy. Ideally chroot
        transports would know enough to cause the external url to be the exact
        one used that caused the chrooting in the first place, but that is not
        currently the case.

        :return: A URL that can be given to another process.
        :raises InProcessTransport: If the transport is one that cannot be
            accessed out of the current process (e.g. a MemoryTransport)
            then InProcessTransport is raised.
        """
        raise NotImplementedError(self.external_url)

    def get_segment_parameters(self):
        """Return the segment parameters for the top segment of the URL.
        """
        return self._segment_parameters

    def set_segment_parameter(self, name, value):
        """Set a segment parameter.

        Args:
          name: Segment parameter name (urlencoded string)
          value: Segment parameter value (urlencoded string)
        """
        if value is None:
            try:
                del self._segment_parameters[name]
            except KeyError:
                pass
        else:
            self._segment_parameters[name] = value
        self.base = urlutils.join_segment_parameters(self._raw_base, self._segment_parameters)

    def _pump(self, from_file, to_file):
        """Most children will need to copy from one file-like
        object or string to another one.
        This just gives them something easy to call.
        """
        return osutils.pumpfile(from_file, to_file)

    def _get_total(self, multi):
        """Try to figure out how many entries are in multi,
        but if not possible, return None.
        """
        try:
            return len(multi)
        except TypeError:
            return None

    def _report_activity(self, bytes, direction):
        """Notify that this transport has activity.

        Implementations should call this from all methods that actually do IO.
        Be careful that it's not called twice, if one method is implemented on
        top of another.

        Args:
          bytes: Number of bytes read or written.
          direction: 'read' or 'write' or None.
        """
        ui.ui_factory.report_transport_activity(self, bytes, direction)

    def _update_pb(self, pb, msg, count, total):
        """Update the progress bar based on the current count
        and total available, total may be None if it was
        not possible to determine.
        """
        if pb is None:
            return
        if total is None:
            pb.update(msg, count, count + 1)
        else:
            pb.update(msg, count, total)

    def _iterate_over(self, multi, func, pb, msg, expand=True):
        """Iterate over all entries in multi, passing them to func,
        and update the progress bar as you go along.

        :param expand:  If True, the entries will be passed to the function
                        by expanding the tuple. If False, it will be passed
                        as a single parameter.
        """
        total = self._get_total(multi)
        result = []
        count = 0
        for entry in multi:
            self._update_pb(pb, msg, count, total)
            if expand:
                result.append(func(*entry))
            else:
                result.append(func(entry))
            count += 1
        return tuple(result)

    def abspath(self, relpath):
        """Return the full url to the given relative path.

        :param relpath: a string of a relative path
        """
        raise NotImplementedError(self.abspath)

    def recommended_page_size(self):
        """Return the recommended page size for this transport.

        This is potentially different for every path in a given namespace.
        For example, local transports might use an operating system call to
        get the block size for a given path, which can vary due to mount
        points.

        :return: The page size in bytes.
        """
        return 4 * 1024

    def relpath(self, abspath):
        """Return the local path portion from a given absolute path.

        This default implementation is not suitable for filesystems with
        aliasing, such as that given by symlinks, where a path may not
        start with our base, but still be a relpath once aliasing is
        resolved.
        """
        if not (abspath == self.base[:-1] or abspath.startswith(self.base)):
            raise errors.PathNotChild(abspath, self.base)
        pl = len(self.base)
        return abspath[pl:].strip('/')

    def local_abspath(self, relpath):
        """Return the absolute path on the local filesystem.

        This function will only be defined for Transports which have a
        physical local filesystem representation.

        :raises errors.NotLocalUrl: When no local path representation is
            available.
        """
        raise errors.NotLocalUrl(self.abspath(relpath))

    def has(self, relpath):
        """Does the file relpath exist?

        Note that some transports MAY allow querying on directories, but this
        is not part of the protocol.  In other words, the results of
        t.has("a_directory_name") are undefined.

        :rtype: bool
        """
        raise NotImplementedError(self.has)

    def has_any(self, relpaths):
        """Return True if any of the paths exist."""
        for relpath in relpaths:
            if self.has(relpath):
                return True
        return False

    def iter_files_recursive(self):
        """Iter the relative paths of files in the transports sub-tree.

        *NOTE*: This only lists *files*, not subdirectories!

        As with other listing functions, only some transports implement this,.
        you may check via listable() to determine if it will.
        """
        raise errors.TransportNotPossible('This transport has not implemented iter_files_recursive (but must claim to be listable to trigger this error).')

    def get(self, relpath):
        """Get the file at the given relative path.

        This may fail in a number of ways:
         - HTTP servers may return content for a directory. (unexpected
           content failure)
         - FTP servers may indicate NoSuchFile for a directory.
         - SFTP servers may give a file handle for a directory that will
           fail on read().

        For correct use of the interface, be sure to catch errors.PathError
        when calling it and catch errors.ReadError when reading from the
        returned object.

        :param relpath: The relative path to the file
        :rtype: File-like object.
        """
        raise NotImplementedError(self.get)

    def get_bytes(self, relpath):
        """Get a raw string of the bytes for a file at the given location.

        :param relpath: The relative path to the file
        """
        f = self.get(relpath)
        try:
            return f.read()
        finally:
            f.close()

    def get_smart_medium(self):
        """Return a smart client medium for this transport if possible.

        A smart medium doesn't imply the presence of a smart server: it implies
        that the smart protocol can be tunnelled via this transport.

        :raises NoSmartMedium: if no smart server medium is available.
        """
        raise errors.NoSmartMedium(self)

    def readv(self, relpath, offsets, adjust_for_latency=False, upper_limit=None):
        """Get parts of the file at the given relative path.

        Args:
          relpath: The path to read data from.
          offsets: A list of (offset, size) tuples.
          adjust_for_latency: Adjust the requested offsets to accomodate
            transport latency. This may re-order the offsets, expand them to
            grab adjacent data when there is likely a high cost to requesting
            data relative to delivering it.
          upper_limit: When adjust_for_latency is True setting upper_limit
            allows the caller to tell the transport about the length of the
            file, so that requests are not issued for ranges beyond the end of
            the file. This matters because some servers and/or transports error
            in such a case rather than just satisfying the available ranges.
            upper_limit should always be provided when adjust_for_latency is
            True, and should be the size of the file in bytes.
        Returns: A list or generator of (offset, data) tuples
        """
        if adjust_for_latency:
            offsets = self._sort_expand_and_combine(offsets, upper_limit)
        return self._readv(relpath, offsets)

    def _readv(self, relpath, offsets):
        """Get parts of the file at the given relative path.

        :param relpath: The path to read.
        :param offsets: A list of (offset, size) tuples.
        :return: A list or generator of (offset, data) tuples
        """
        if not offsets:
            return
        fp = self.get(relpath)
        return self._seek_and_read(fp, offsets, relpath)

    def _seek_and_read(self, fp, offsets, relpath='<unknown>'):
        """An implementation of readv that uses fp.seek and fp.read.

        This uses _coalesce_offsets to issue larger reads and fewer seeks.

        :param fp: A file-like object that supports seek() and read(size).
            Note that implementations are allowed to call .close() on this file
            handle, so don't trust that you can use it for other work.
        :param offsets: A list of offsets to be read from the given file.
        :return: yield (pos, data) tuples for each request
        """
        offsets = list(offsets)
        sorted_offsets = sorted(offsets)
        offset_stack = iter(offsets)
        cur_offset_and_size = next(offset_stack)
        coalesced = self._coalesce_offsets(sorted_offsets, limit=self._max_readv_combine, fudge_factor=self._bytes_to_read_before_seek)
        data_map = {}
        try:
            for c_offset in coalesced:
                fp.seek(c_offset.start)
                data = fp.read(c_offset.length)
                if len(data) < c_offset.length:
                    raise errors.ShortReadvError(relpath, c_offset.start, c_offset.length, actual=len(data))
                for suboffset, subsize in c_offset.ranges:
                    key = (c_offset.start + suboffset, subsize)
                    data_map[key] = data[suboffset:suboffset + subsize]
                while cur_offset_and_size in data_map:
                    this_data = data_map.pop(cur_offset_and_size)
                    this_offset = cur_offset_and_size[0]
                    try:
                        cur_offset_and_size = next(offset_stack)
                    except StopIteration:
                        fp.close()
                        cur_offset_and_size = None
                    yield (this_offset, this_data)
        finally:
            fp.close()

    def _sort_expand_and_combine(self, offsets, upper_limit):
        """Helper for readv.

        :param offsets: A readv vector - (offset, length) tuples.
        :param upper_limit: The highest byte offset that may be requested.
        :return: A readv vector that will read all the regions requested by
            offsets, in start-to-end order, with no duplicated regions,
            expanded by the transports recommended page size.
        """
        offsets = sorted(offsets)
        if len(offsets) == 0:

            def empty_yielder():
                if False:
                    yield None
            return empty_yielder()
        maximum_expansion = self.recommended_page_size()
        new_offsets = []
        for offset, length in offsets:
            expansion = maximum_expansion - length
            if expansion < 0:
                expansion = 0
            reduction = expansion // 2
            new_offset = offset - reduction
            new_length = length + expansion
            if new_offset < 0:
                new_offset = 0
            if upper_limit is not None and new_offset + new_length > upper_limit:
                new_length = upper_limit - new_offset
            new_offsets.append((new_offset, new_length))
        offsets = []
        current_offset, current_length = new_offsets[0]
        current_finish = current_length + current_offset
        for offset, length in new_offsets[1:]:
            finish = offset + length
            if offset > current_finish:
                offsets.append((current_offset, current_length))
                current_offset = offset
                current_length = length
                current_finish = finish
                continue
            if finish > current_finish:
                current_finish = finish
                current_length = finish - current_offset
        offsets.append((current_offset, current_length))
        return offsets

    @staticmethod
    def _coalesce_offsets(offsets, limit=0, fudge_factor=0, max_size=0):
        """Yield coalesced offsets.

        With a long list of neighboring requests, combine them
        into a single large request, while retaining the original
        offsets.
        Turns  [(15, 10), (25, 10)] => [(15, 20, [(0, 10), (10, 10)])]
        Note that overlapping requests are not permitted. (So [(15, 10), (20,
        10)] will raise a ValueError.) This is because the data we access never
        overlaps, and it allows callers to trust that we only need any byte of
        data for 1 request (so nothing needs to be buffered to fulfill a second
        request.)

        :param offsets: A list of (start, length) pairs
        :param limit: Only combine a maximum of this many pairs Some transports
                penalize multiple reads more than others, and sometimes it is
                better to return early.
                0 means no limit
        :param fudge_factor: All transports have some level of 'it is
                better to read some more data and throw it away rather
                than seek', so collapse if we are 'close enough'
        :param max_size: Create coalesced offsets no bigger than this size.
                When a single offset is bigger than 'max_size', it will keep
                its size and be alone in the coalesced offset.
                0 means no maximum size.
        :return: return a list of _CoalescedOffset objects, which have members
            for where to start, how much to read, and how to split those chunks
            back up
        """
        last_end = None
        cur = _CoalescedOffset(None, None, [])
        coalesced_offsets = []
        if max_size <= 0:
            max_size = 100 * 1024 * 1024
        for start, size in offsets:
            end = start + size
            if last_end is not None and start <= last_end + fudge_factor and (start >= cur.start) and (limit <= 0 or len(cur.ranges) < limit) and (max_size <= 0 or end - cur.start <= max_size):
                if start < last_end:
                    raise ValueError('Overlapping range not allowed: last range ended at %s, new one starts at %s' % (last_end, start))
                cur.length = end - cur.start
                cur.ranges.append((start - cur.start, size))
            else:
                if cur.start is not None:
                    coalesced_offsets.append(cur)
                cur = _CoalescedOffset(start, size, [(0, size)])
            last_end = end
        if cur.start is not None:
            coalesced_offsets.append(cur)
        return coalesced_offsets

    def put_bytes(self, relpath: str, raw_bytes: bytes, mode=None):
        """Atomically put the supplied bytes into the given location.

        :param relpath: The location to put the contents, relative to the
            transport base.
        :param raw_bytes: A bytestring of data.
        :param mode: Create the file with the given mode.
        :return: None
        """
        if not isinstance(raw_bytes, bytes):
            raise TypeError('raw_bytes must be a plain string, not %s' % type(raw_bytes))
        return self.put_file(relpath, BytesIO(raw_bytes), mode=mode)

    def put_bytes_non_atomic(self, relpath, raw_bytes: bytes, mode=None, create_parent_dir=False, dir_mode=None):
        """Copy the string into the target location.

        This function is not strictly safe to use. See
        Transport.put_bytes_non_atomic for more information.

        :param relpath: The remote location to put the contents.
        :param raw_bytes:   A string object containing the raw bytes to write
                        into the target file.
        :param mode:    Possible access permissions for new file.
                        None means do not set remote permissions.
        :param create_parent_dir: If we cannot create the target file because
                        the parent directory does not exist, go ahead and
                        create it, and then try again.
        :param dir_mode: Possible access permissions for new directories.
        """
        if not isinstance(raw_bytes, bytes):
            raise TypeError('raw_bytes must be a plain string, not %s' % type(raw_bytes))
        self.put_file_non_atomic(relpath, BytesIO(raw_bytes), mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)

    def put_file(self, relpath, f, mode=None):
        """Copy the file-like object into the location.

        :param relpath: Location to put the contents, relative to base.
        :param f:       File-like object.
        :param mode: The mode for the newly created file,
                     None means just use the default.
        :return: The length of the file that was written.
        """
        raise NotImplementedError(self.put_file)

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
        :param dir_mode: Possible access permissions for new directories.
        """
        try:
            return self.put_file(relpath, f, mode=mode)
        except NoSuchFile:
            if not create_parent_dir:
                raise
            parent_dir = osutils.dirname(relpath)
            if parent_dir:
                self.mkdir(parent_dir, mode=dir_mode)
                return self.put_file(relpath, f, mode=mode)

    def mkdir(self, relpath, mode=None):
        """Create a directory at the given path."""
        raise NotImplementedError(self.mkdir)

    def open_write_stream(self, relpath, mode=None):
        """Open a writable file stream at relpath.

        A file stream is a file like object with a write() method that accepts
        bytes to write.. Buffering may occur internally until the stream is
        closed with stream.close().  Calls to readv or the get_* methods will
        be synchronised with any internal buffering that may be present.

        :param relpath: The relative path to the file.
        :param mode: The mode for the newly created file,
                     None means just use the default
        :return: A FileStream. FileStream objects have two methods, write() and
            close(). There is no guarantee that data is committed to the file
            if close() has not been called (even if get() is called on the same
            path).
        """
        raise NotImplementedError(self.open_write_stream)

    def append_file(self, relpath, f, mode=None):
        """Append bytes from a file-like object to a file at relpath.

        The file is created if it does not already exist.

        :param f: a file-like object of the bytes to append.
        :param mode: Unix mode for newly created files.  This is not used for
            existing files.

        :returns: the length of relpath before the content was written to it.
        """
        raise NotImplementedError(self.append_file)

    def append_bytes(self, relpath, data, mode=None):
        """Append bytes to a file at relpath.

        The file is created if it does not already exist.

        :param relpath: The relative path to the file.
        :param data: a string of the bytes to append.
        :param mode: Unix mode for newly created files.  This is not used for
            existing files.

        :returns: the length of relpath before the content was written to it.
        """
        if not isinstance(data, bytes):
            raise TypeError('bytes must be a plain string, not %s' % type(data))
        return self.append_file(relpath, BytesIO(data), mode=mode)

    def copy(self, rel_from, rel_to):
        """Copy the item at rel_from to the location at rel_to.

        Override this for efficiency if a specific transport can do it
        faster than this default implementation.
        """
        with self.get(rel_from) as f:
            self.put_file(rel_to, f)

    def copy_to(self, relpaths, other, mode=None, pb=None):
        """Copy a set of entries from self into another Transport.

        :param relpaths: A list/generator of entries to be copied.
        :param mode: This is the target mode for the newly created files
        TODO: This interface needs to be updated so that the target location
              can be different from the source location.
        """

        def copy_entry(path):
            other.put_file(path, self.get(path), mode=mode)
        return len(self._iterate_over(relpaths, copy_entry, pb, 'copy_to', expand=False))

    def copy_tree(self, from_relpath, to_relpath):
        """Copy a subtree from one relpath to another.

        If a faster implementation is available, specific transports should
        implement it.
        """
        source = self.clone(from_relpath)
        target = self.clone(to_relpath)
        stat = self.stat(from_relpath)
        target.mkdir('.', stat.st_mode & 511)
        source.copy_tree_to_transport(target)

    def copy_tree_to_transport(self, to_transport):
        """Copy a subtree from one transport to another.

        self.base is used as the source tree root, and to_transport.base
        is used as the target.  to_transport.base must exist (and be a
        directory).
        """
        files = []
        directories = ['.']
        while directories:
            dir = directories.pop()
            if dir != '.':
                to_transport.mkdir(dir)
            for path in self.list_dir(dir):
                path = dir + '/' + path
                stat = self.stat(path)
                if S_ISDIR(stat.st_mode):
                    directories.append(path)
                else:
                    files.append(path)
        self.copy_to(files, to_transport)

    def rename(self, rel_from, rel_to):
        """Rename a file or directory.

        This *must* fail if the destination is a nonempty directory - it must
        not automatically remove it.  It should raise DirectoryNotEmpty, or
        some other PathError if the case can't be specifically detected.

        If the destination is an empty directory or a file this function may
        either fail or succeed, depending on the underlying transport.  It
        should not attempt to remove the destination if overwriting is not the
        native transport behaviour.  If at all possible the transport should
        ensure that the rename either completes or not, without leaving the
        destination deleted and the new file not moved in place.

        This is intended mainly for use in implementing LockDir.
        """
        raise NotImplementedError(self.rename)

    def move(self, rel_from, rel_to):
        """Move the item at rel_from to the location at rel_to.

        The destination is deleted if possible, even if it's a non-empty
        directory tree.

        If a transport can directly implement this it is suggested that
        it do so for efficiency.
        """
        if S_ISDIR(self.stat(rel_from).st_mode):
            self.copy_tree(rel_from, rel_to)
            self.delete_tree(rel_from)
        else:
            self.copy(rel_from, rel_to)
            self.delete(rel_from)

    def delete(self, relpath):
        """Delete the item at relpath"""
        raise NotImplementedError(self.delete)

    def delete_tree(self, relpath):
        """Delete an entire tree. This may require a listable transport."""
        subtree = self.clone(relpath)
        files = []
        directories = ['.']
        pending_rmdirs = []
        while directories:
            dir = directories.pop()
            if dir != '.':
                pending_rmdirs.append(dir)
            for path in subtree.list_dir(dir):
                path = dir + '/' + path
                stat = subtree.stat(path)
                if S_ISDIR(stat.st_mode):
                    directories.append(path)
                else:
                    files.append(path)
        for file in files:
            subtree.delete(file)
        pending_rmdirs.reverse()
        for dir in pending_rmdirs:
            subtree.rmdir(dir)
        self.rmdir(relpath)

    def __repr__(self):
        return '<{}.{} url={}>'.format(self.__module__, self.__class__.__name__, self.base)

    def stat(self, relpath):
        """Return the stat information for a file.
        WARNING: This may not be implementable for all protocols, so use
        sparingly.
        NOTE: This returns an object with fields such as 'st_size'. It MAY
        or MAY NOT return the literal result of an os.stat() call, so all
        access should be via named fields.
        ALSO NOTE: Stats of directories may not be supported on some
        transports.
        """
        raise NotImplementedError(self.stat)

    def rmdir(self, relpath):
        """Remove a directory at the given path."""
        raise NotImplementedError

    def readlink(self, relpath):
        """Return a string representing the path to which the symbolic link points."""
        raise errors.TransportNotPossible('Dereferencing symlinks is not supported on %s' % self)

    def hardlink(self, source, link_name):
        """Create a hardlink pointing to source named link_name."""
        raise errors.TransportNotPossible('Hard links are not supported on %s' % self)

    def symlink(self, source, link_name):
        """Create a symlink pointing to source named link_name."""
        raise errors.TransportNotPossible('Symlinks are not supported on %s' % self)

    def listable(self):
        """Return True if this store supports listing."""
        raise NotImplementedError(self.listable)

    def list_dir(self, relpath):
        """Return a list of all files at the given location.
        WARNING: many transports do not support this, so trying avoid using
        it if at all possible.
        """
        raise errors.TransportNotPossible('Transport %r has not implemented list_dir (but must claim to be listable to trigger this error).' % self)

    def lock_read(self, relpath):
        """Lock the given file for shared (read) access.

        WARNING: many transports do not support this, so trying avoid using it.
        These methods may be removed in the future.

        Transports may raise TransportNotPossible if OS-level locks cannot be
        taken over this transport.

        :return: A lock object, which should contain an unlock() function.
        """
        raise errors.TransportNotPossible('transport locks not supported on %s' % self)

    def lock_write(self, relpath):
        """Lock the given file for exclusive (write) access.

        WARNING: many transports do not support this, so trying avoid using it.
        These methods may be removed in the future.

        Transports may raise TransportNotPossible if OS-level locks cannot be
        taken over this transport.

        :return: A lock object, which should contain an unlock() function.
        """
        raise errors.TransportNotPossible('transport locks not supported on %s' % self)

    def is_readonly(self):
        """Return true if this connection cannot be written to."""
        return False

    def _can_roundtrip_unix_modebits(self):
        """Return true if this transport can store and retrieve unix modebits.

        (For example, 0700 to make a directory owner-private.)

        Note: most callers will not want to switch on this, but should rather
        just try and set permissions and let them be either stored or not.
        This is intended mainly for the use of the test suite.

        Warning: this is not guaranteed to be accurate as sometimes we can't
        be sure: for example with vfat mounted on unix, or a windows sftp
        server."""
        return False

    def _reuse_for(self, other_base):
        return None

    def disconnect(self):
        pass

    def _redirected_to(self, source, target):
        """Returns a transport suitable to re-issue a redirected request.

        :param source: The source url as returned by the server.
        :param target: The target url as returned by the server.

        The redirection can be handled only if the relpath involved is not
        renamed by the redirection.

        :returns: A transport
        :raise UnusableRedirect: when redirection can not be provided
        """
        raise UnusableRedirect(source, target, 'transport does not support redirection')