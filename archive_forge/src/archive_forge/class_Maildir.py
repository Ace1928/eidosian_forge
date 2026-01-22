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
class Maildir(Mailbox):
    """A qmail-style Maildir mailbox."""
    colon = ':'

    def __init__(self, dirname, factory=None, create=True):
        """Initialize a Maildir instance."""
        Mailbox.__init__(self, dirname, factory, create)
        self._paths = {'tmp': os.path.join(self._path, 'tmp'), 'new': os.path.join(self._path, 'new'), 'cur': os.path.join(self._path, 'cur')}
        if not os.path.exists(self._path):
            if create:
                os.mkdir(self._path, 448)
                for path in self._paths.values():
                    os.mkdir(path, 448)
            else:
                raise NoSuchMailboxError(self._path)
        self._toc = {}
        self._toc_mtimes = {'cur': 0, 'new': 0}
        self._last_read = 0
        self._skewfactor = 0.1

    def add(self, message):
        """Add message and return assigned key."""
        tmp_file = self._create_tmp()
        try:
            self._dump_message(message, tmp_file)
        except BaseException:
            tmp_file.close()
            os.remove(tmp_file.name)
            raise
        _sync_close(tmp_file)
        if isinstance(message, MaildirMessage):
            subdir = message.get_subdir()
            suffix = self.colon + message.get_info()
            if suffix == self.colon:
                suffix = ''
        else:
            subdir = 'new'
            suffix = ''
        uniq = os.path.basename(tmp_file.name).split(self.colon)[0]
        dest = os.path.join(self._path, subdir, uniq + suffix)
        if isinstance(message, MaildirMessage):
            os.utime(tmp_file.name, (os.path.getatime(tmp_file.name), message.get_date()))
        try:
            try:
                os.link(tmp_file.name, dest)
            except (AttributeError, PermissionError):
                os.rename(tmp_file.name, dest)
            else:
                os.remove(tmp_file.name)
        except OSError as e:
            os.remove(tmp_file.name)
            if e.errno == errno.EEXIST:
                raise ExternalClashError('Name clash with existing message: %s' % dest)
            else:
                raise
        return uniq

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        os.remove(os.path.join(self._path, self._lookup(key)))

    def discard(self, key):
        """If the keyed message exists, remove it."""
        try:
            self.remove(key)
        except (KeyError, FileNotFoundError):
            pass

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        old_subpath = self._lookup(key)
        temp_key = self.add(message)
        temp_subpath = self._lookup(temp_key)
        if isinstance(message, MaildirMessage):
            dominant_subpath = temp_subpath
        else:
            dominant_subpath = old_subpath
        subdir = os.path.dirname(dominant_subpath)
        if self.colon in dominant_subpath:
            suffix = self.colon + dominant_subpath.split(self.colon)[-1]
        else:
            suffix = ''
        self.discard(key)
        tmp_path = os.path.join(self._path, temp_subpath)
        new_path = os.path.join(self._path, subdir, key + suffix)
        if isinstance(message, MaildirMessage):
            os.utime(tmp_path, (os.path.getatime(tmp_path), message.get_date()))
        os.rename(tmp_path, new_path)

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        subpath = self._lookup(key)
        with open(os.path.join(self._path, subpath), 'rb') as f:
            if self._factory:
                msg = self._factory(f)
            else:
                msg = MaildirMessage(f)
        subdir, name = os.path.split(subpath)
        msg.set_subdir(subdir)
        if self.colon in name:
            msg.set_info(name.split(self.colon)[-1])
        msg.set_date(os.path.getmtime(os.path.join(self._path, subpath)))
        return msg

    def get_bytes(self, key):
        """Return a bytes representation or raise a KeyError."""
        with open(os.path.join(self._path, self._lookup(key)), 'rb') as f:
            return f.read().replace(linesep, b'\n')

    def get_file(self, key):
        """Return a file-like representation or raise a KeyError."""
        f = open(os.path.join(self._path, self._lookup(key)), 'rb')
        return _ProxyFile(f)

    def iterkeys(self):
        """Return an iterator over keys."""
        self._refresh()
        for key in self._toc:
            try:
                self._lookup(key)
            except KeyError:
                continue
            yield key

    def __contains__(self, key):
        """Return True if the keyed message exists, False otherwise."""
        self._refresh()
        return key in self._toc

    def __len__(self):
        """Return a count of messages in the mailbox."""
        self._refresh()
        return len(self._toc)

    def flush(self):
        """Write any pending changes to disk."""
        pass

    def lock(self):
        """Lock the mailbox."""
        return

    def unlock(self):
        """Unlock the mailbox if it is locked."""
        return

    def close(self):
        """Flush and close the mailbox."""
        return

    def list_folders(self):
        """Return a list of folder names."""
        result = []
        for entry in os.listdir(self._path):
            if len(entry) > 1 and entry[0] == '.' and os.path.isdir(os.path.join(self._path, entry)):
                result.append(entry[1:])
        return result

    def get_folder(self, folder):
        """Return a Maildir instance for the named folder."""
        return Maildir(os.path.join(self._path, '.' + folder), factory=self._factory, create=False)

    def add_folder(self, folder):
        """Create a folder and return a Maildir instance representing it."""
        path = os.path.join(self._path, '.' + folder)
        result = Maildir(path, factory=self._factory)
        maildirfolder_path = os.path.join(path, 'maildirfolder')
        if not os.path.exists(maildirfolder_path):
            os.close(os.open(maildirfolder_path, os.O_CREAT | os.O_WRONLY, 438))
        return result

    def remove_folder(self, folder):
        """Delete the named folder, which must be empty."""
        path = os.path.join(self._path, '.' + folder)
        for entry in os.listdir(os.path.join(path, 'new')) + os.listdir(os.path.join(path, 'cur')):
            if len(entry) < 1 or entry[0] != '.':
                raise NotEmptyError('Folder contains message(s): %s' % folder)
        for entry in os.listdir(path):
            if entry != 'new' and entry != 'cur' and (entry != 'tmp') and os.path.isdir(os.path.join(path, entry)):
                raise NotEmptyError("Folder contains subdirectory '%s': %s" % (folder, entry))
        for root, dirs, files in os.walk(path, topdown=False):
            for entry in files:
                os.remove(os.path.join(root, entry))
            for entry in dirs:
                os.rmdir(os.path.join(root, entry))
        os.rmdir(path)

    def clean(self):
        """Delete old files in "tmp"."""
        now = time.time()
        for entry in os.listdir(os.path.join(self._path, 'tmp')):
            path = os.path.join(self._path, 'tmp', entry)
            if now - os.path.getatime(path) > 129600:
                os.remove(path)
    _count = 1

    def _create_tmp(self):
        """Create a file in the tmp subdirectory and open and return it."""
        now = time.time()
        hostname = socket.gethostname()
        if '/' in hostname:
            hostname = hostname.replace('/', '\\057')
        if ':' in hostname:
            hostname = hostname.replace(':', '\\072')
        uniq = '%s.M%sP%sQ%s.%s' % (int(now), int(now % 1 * 1000000.0), os.getpid(), Maildir._count, hostname)
        path = os.path.join(self._path, 'tmp', uniq)
        try:
            os.stat(path)
        except FileNotFoundError:
            Maildir._count += 1
            try:
                return _create_carefully(path)
            except FileExistsError:
                pass
        raise ExternalClashError('Name clash prevented file creation: %s' % path)

    def _refresh(self):
        """Update table of contents mapping."""
        if time.time() - self._last_read > 2 + self._skewfactor:
            refresh = False
            for subdir in self._toc_mtimes:
                mtime = os.path.getmtime(self._paths[subdir])
                if mtime > self._toc_mtimes[subdir]:
                    refresh = True
                self._toc_mtimes[subdir] = mtime
            if not refresh:
                return
        self._toc = {}
        for subdir in self._toc_mtimes:
            path = self._paths[subdir]
            for entry in os.listdir(path):
                p = os.path.join(path, entry)
                if os.path.isdir(p):
                    continue
                uniq = entry.split(self.colon)[0]
                self._toc[uniq] = os.path.join(subdir, entry)
        self._last_read = time.time()

    def _lookup(self, key):
        """Use TOC to return subpath for given key, or raise a KeyError."""
        try:
            if os.path.exists(os.path.join(self._path, self._toc[key])):
                return self._toc[key]
        except KeyError:
            pass
        self._refresh()
        try:
            return self._toc[key]
        except KeyError:
            raise KeyError('No message with key: %s' % key) from None

    def next(self):
        """Return the next message in a one-time iteration."""
        if not hasattr(self, '_onetime_keys'):
            self._onetime_keys = self.iterkeys()
        while True:
            try:
                return self[next(self._onetime_keys)]
            except StopIteration:
                return None
            except KeyError:
                continue