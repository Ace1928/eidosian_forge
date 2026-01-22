import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
class _singlefileMailbox(Mailbox):
    """A single-file mailbox."""

    def __init__(self, path, factory=None, create=True):
        """Initialize a single-file mailbox."""
        Mailbox.__init__(self, path, factory, create)
        try:
            f = open(self._path, 'rb+')
        except OSError as e:
            if e.errno == errno.ENOENT:
                if create:
                    f = open(self._path, 'wb+')
                else:
                    raise NoSuchMailboxError(self._path)
            elif e.errno in (errno.EACCES, errno.EROFS):
                f = open(self._path, 'rb')
            else:
                raise
        self._file = f
        self._toc = None
        self._next_key = 0
        self._pending = False
        self._pending_sync = False
        self._locked = False
        self._file_length = None

    def add(self, message):
        """Add message and return assigned key."""
        self._lookup()
        self._toc[self._next_key] = self._append_message(message)
        self._next_key += 1
        self._pending_sync = True
        return self._next_key - 1

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        self._lookup(key)
        del self._toc[key]
        self._pending = True

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        self._lookup(key)
        self._toc[key] = self._append_message(message)
        self._pending = True

    def iterkeys(self):
        """Return an iterator over keys."""
        self._lookup()
        yield from self._toc.keys()

    def __contains__(self, key):
        """Return True if the keyed message exists, False otherwise."""
        self._lookup()
        return key in self._toc

    def __len__(self):
        """Return a count of messages in the mailbox."""
        self._lookup()
        return len(self._toc)

    def lock(self):
        """Lock the mailbox."""
        if not self._locked:
            _lock_file(self._file)
            self._locked = True

    def unlock(self):
        """Unlock the mailbox if it is locked."""
        if self._locked:
            _unlock_file(self._file)
            self._locked = False

    def flush(self):
        """Write any pending changes to disk."""
        if not self._pending:
            if self._pending_sync:
                _sync_flush(self._file)
                self._pending_sync = False
            return
        assert self._toc is not None
        self._file.seek(0, 2)
        cur_len = self._file.tell()
        if cur_len != self._file_length:
            raise ExternalClashError('Size of mailbox file changed (expected %i, found %i)' % (self._file_length, cur_len))
        new_file = _create_temporary(self._path)
        try:
            new_toc = {}
            self._pre_mailbox_hook(new_file)
            for key in sorted(self._toc.keys()):
                start, stop = self._toc[key]
                self._file.seek(start)
                self._pre_message_hook(new_file)
                new_start = new_file.tell()
                while True:
                    buffer = self._file.read(min(4096, stop - self._file.tell()))
                    if not buffer:
                        break
                    new_file.write(buffer)
                new_toc[key] = (new_start, new_file.tell())
                self._post_message_hook(new_file)
            self._file_length = new_file.tell()
        except:
            new_file.close()
            os.remove(new_file.name)
            raise
        _sync_close(new_file)
        self._file.close()
        mode = os.stat(self._path).st_mode
        os.chmod(new_file.name, mode)
        try:
            os.rename(new_file.name, self._path)
        except FileExistsError:
            os.remove(self._path)
            os.rename(new_file.name, self._path)
        self._file = open(self._path, 'rb+')
        self._toc = new_toc
        self._pending = False
        self._pending_sync = False
        if self._locked:
            _lock_file(self._file, dotlock=False)

    def _pre_mailbox_hook(self, f):
        """Called before writing the mailbox to file f."""
        return

    def _pre_message_hook(self, f):
        """Called before writing each message to file f."""
        return

    def _post_message_hook(self, f):
        """Called after writing each message to file f."""
        return

    def close(self):
        """Flush and close the mailbox."""
        try:
            self.flush()
        finally:
            try:
                if self._locked:
                    self.unlock()
            finally:
                self._file.close()

    def _lookup(self, key=None):
        """Return (start, stop) or raise KeyError."""
        if self._toc is None:
            self._generate_toc()
        if key is not None:
            try:
                return self._toc[key]
            except KeyError:
                raise KeyError('No message with key: %s' % key) from None

    def _append_message(self, message):
        """Append message to mailbox and return (start, stop) offsets."""
        self._file.seek(0, 2)
        before = self._file.tell()
        if len(self._toc) == 0 and (not self._pending):
            self._pre_mailbox_hook(self._file)
        try:
            self._pre_message_hook(self._file)
            offsets = self._install_message(message)
            self._post_message_hook(self._file)
        except BaseException:
            self._file.truncate(before)
            raise
        self._file.flush()
        self._file_length = self._file.tell()
        return offsets