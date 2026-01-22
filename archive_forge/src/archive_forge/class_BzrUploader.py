from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
class BzrUploader:

    def __init__(self, branch, to_transport, outf, tree, rev_id, quiet=False):
        self.branch = branch
        self.to_transport = to_transport
        self.outf = outf
        self.tree = tree
        self.rev_id = rev_id
        self.quiet = quiet
        self._pending_deletions = []
        self._pending_renames = []
        self._uploaded_revid = None
        self._ignored = None

    def _up_stat(self, relpath):
        return self.to_transport.stat(urlutils.escape(relpath))

    def _up_rename(self, old_path, new_path):
        return self.to_transport.rename(urlutils.escape(old_path), urlutils.escape(new_path))

    def _up_delete(self, relpath):
        return self.to_transport.delete(urlutils.escape(relpath))

    def _up_delete_tree(self, relpath):
        return self.to_transport.delete_tree(urlutils.escape(relpath))

    def _up_mkdir(self, relpath, mode):
        return self.to_transport.mkdir(urlutils.escape(relpath), mode)

    def _up_rmdir(self, relpath):
        return self.to_transport.rmdir(urlutils.escape(relpath))

    def _up_put_bytes(self, relpath, bytes, mode):
        self.to_transport.put_bytes(urlutils.escape(relpath), bytes, mode)

    def _up_get_bytes(self, relpath):
        return self.to_transport.get_bytes(urlutils.escape(relpath))

    def set_uploaded_revid(self, rev_id):
        revid_path = self.branch.get_config_stack().get('upload_revid_location')
        self.to_transport.put_bytes(urlutils.escape(revid_path), rev_id)
        self._uploaded_revid = rev_id

    def get_uploaded_revid(self):
        if self._uploaded_revid is None:
            revid_path = self.branch.get_config_stack().get('upload_revid_location')
            try:
                self._uploaded_revid = self._up_get_bytes(revid_path)
            except transport.NoSuchFile:
                self._uploaded_revid = revision.NULL_REVISION
        return self._uploaded_revid

    def _get_ignored(self):
        if self._ignored is None:
            try:
                ignore_file_path = '.bzrignore-upload'
                ignore_file = self.tree.get_file(ignore_file_path)
            except transport.NoSuchFile:
                ignored_patterns = []
            else:
                ignored_patterns = ignores.parse_ignore_file(ignore_file)
            self._ignored = globbing.Globster(ignored_patterns)
        return self._ignored

    def is_ignored(self, relpath):
        glob = self._get_ignored()
        ignored = glob.match(relpath)
        import os
        if not ignored:
            dir = os.path.dirname(relpath)
            while dir and (not ignored):
                ignored = glob.match(dir)
                if not ignored:
                    dir = os.path.dirname(dir)
        return ignored

    def upload_file(self, old_relpath, new_relpath, mode=None):
        if mode is None:
            if self.tree.is_executable(new_relpath):
                mode = 509
            else:
                mode = 436
        if not self.quiet:
            self.outf.write('Uploading %s\n' % old_relpath)
        self._up_put_bytes(old_relpath, self.tree.get_file_text(new_relpath), mode)

    def _force_clear(self, relpath):
        try:
            st = self._up_stat(relpath)
            if stat.S_ISDIR(st.st_mode):
                if not self.quiet:
                    self.outf.write('Clearing {}/{}\n'.format(self.to_transport.external_url(), relpath))
                self._up_delete_tree(relpath)
            elif stat.S_ISLNK(st.st_mode):
                if not self.quiet:
                    self.outf.write('Clearing {}/{}\n'.format(self.to_transport.external_url(), relpath))
                self._up_delete(relpath)
        except errors.PathError:
            pass

    def upload_file_robustly(self, relpath, mode=None):
        """Upload a file, clearing the way on the remote side.

        When doing a full upload, it may happen that a directory exists where
        we want to put our file.
        """
        self._force_clear(relpath)
        self.upload_file(relpath, relpath, mode)

    def upload_symlink(self, relpath, target):
        self.to_transport.symlink(target, relpath)

    def upload_symlink_robustly(self, relpath, target):
        """Handle uploading symlinks.
        """
        self._force_clear(relpath)
        target = osutils.normpath(osutils.pathjoin(osutils.dirname(relpath), target))
        self.upload_symlink(relpath, target)

    def make_remote_dir(self, relpath, mode=None):
        if mode is None:
            mode = 509
        self._up_mkdir(relpath, mode)

    def make_remote_dir_robustly(self, relpath, mode=None):
        """Create a remote directory, clearing the way on the remote side.

        When doing a full upload, it may happen that a file exists where we
        want to create our directory.
        """
        try:
            st = self._up_stat(relpath)
            if not stat.S_ISDIR(st.st_mode):
                if not self.quiet:
                    self.outf.write('Deleting {}/{}\n'.format(self.to_transport.external_url(), relpath))
                self._up_delete(relpath)
            else:
                return
        except errors.PathError:
            pass
        self.make_remote_dir(relpath, mode)

    def delete_remote_file(self, relpath):
        if not self.quiet:
            self.outf.write('Deleting %s\n' % relpath)
        self._up_delete(relpath)

    def delete_remote_dir(self, relpath):
        if not self.quiet:
            self.outf.write('Deleting %s\n' % relpath)
        self._up_rmdir(relpath)

    def delete_remote_dir_maybe(self, relpath):
        """Try to delete relpath, keeping failures to retry later."""
        try:
            self._up_rmdir(relpath)
        except errors.PathError:
            self._pending_deletions.append(relpath)

    def finish_deletions(self):
        if self._pending_deletions:
            for relpath in reversed(self._pending_deletions):
                self._up_rmdir(relpath)
            self._pending_deletions = []

    def rename_remote(self, old_relpath, new_relpath):
        """Rename a remote file or directory taking care of collisions.

        To avoid collisions during bulk renames, each renamed target is
        temporarily assigned a unique name. When all renames have been done,
        each target get its proper name.
        """
        import os
        import random
        import time
        stamp = '.tmp.%.9f.%d.%d' % (time.time(), os.getpid(), random.randint(0, 2147483647))
        if not self.quiet:
            self.outf.write('Renaming {} to {}\n'.format(old_relpath, new_relpath))
        self._up_rename(old_relpath, stamp)
        self._pending_renames.append((stamp, new_relpath))

    def finish_renames(self):
        for stamp, new_path in self._pending_renames:
            self._up_rename(stamp, new_path)
        self._pending_renames = []

    def upload_full_tree(self):
        self.to_transport.ensure_base()
        with self.tree.lock_read():
            for relpath, ie in self.tree.iter_entries_by_dir():
                if relpath in ('', '.bzrignore', '.bzrignore-upload'):
                    continue
                if self.is_ignored(relpath):
                    if not self.quiet:
                        self.outf.write('Ignoring %s\n' % relpath)
                    continue
                if ie.kind == 'file':
                    self.upload_file_robustly(relpath)
                elif ie.kind == 'symlink':
                    try:
                        self.upload_symlink_robustly(relpath, ie.symlink_target)
                    except errors.TransportNotPossible:
                        if not self.quiet:
                            target = self.tree.path_content_summary(relpath)[3]
                            self.outf.write('Not uploading symlink %s -> %s\n' % (relpath, target))
                elif ie.kind == 'directory':
                    self.make_remote_dir_robustly(relpath)
                else:
                    raise NotImplementedError
            self.set_uploaded_revid(self.rev_id)

    def upload_tree(self):
        rev_id = self.get_uploaded_revid()
        if rev_id == revision.NULL_REVISION:
            if not self.quiet:
                self.outf.write('No uploaded revision id found, switching to full upload\n')
            self.upload_full_tree()
            return
        if rev_id == self.rev_id:
            if not self.quiet:
                self.outf.write('Remote location already up to date\n')
        from_tree = self.branch.repository.revision_tree(rev_id)
        self.to_transport.ensure_base()
        changes = self.tree.changes_from(from_tree)
        with self.tree.lock_read():
            for change in changes.removed:
                if self.is_ignored(change.path[0]):
                    if not self.quiet:
                        self.outf.write('Ignoring %s\n' % change.path[0])
                    continue
                if change.kind[0] == 'file':
                    self.delete_remote_file(change.path[0])
                elif change.kind[0] == 'directory':
                    self.delete_remote_dir_maybe(change.path[0])
                elif change.kind[0] == 'symlink':
                    self.delete_remote_file(change.path[0])
                else:
                    raise NotImplementedError
            for change in changes.renamed:
                if self.is_ignored(change.path[0]) and self.is_ignored(change.path[1]):
                    if not self.quiet:
                        self.outf.write('Ignoring %s\n' % change.path[0])
                        self.outf.write('Ignoring %s\n' % change.path[1])
                    continue
                if change.changed_content:
                    self.upload_file(change.path[0], change.path[1])
                self.rename_remote(change.path[0], change.path[1])
            self.finish_renames()
            self.finish_deletions()
            for change in changes.kind_changed:
                if self.is_ignored(change.path[1]):
                    if not self.quiet:
                        self.outf.write('Ignoring %s\n' % change.path[1])
                    continue
                if change.kind[0] in ('file', 'symlink'):
                    self.delete_remote_file(change.path[0])
                elif change.kind[0] == 'directory':
                    self.delete_remote_dir(change.path[0])
                else:
                    raise NotImplementedError
                if change.kind[1] == 'file':
                    self.upload_file(change.path[1], change.path[1])
                elif change.kind[1] == 'symlink':
                    target = self.tree.get_symlink_target(change.path[1])
                    self.upload_symlink(change.path[1], target)
                elif change.kind[1] == 'directory':
                    self.make_remote_dir(change.path[1])
                else:
                    raise NotImplementedError
            for change in changes.added + changes.copied:
                if self.is_ignored(change.path[1]):
                    if not self.quiet:
                        self.outf.write('Ignoring %s\n' % change.path[1])
                    continue
                if change.kind[1] == 'file':
                    self.upload_file(change.path[1], change.path[1])
                elif change.kind[1] == 'directory':
                    self.make_remote_dir(change.path[1])
                elif change.kind[1] == 'symlink':
                    target = self.tree.get_symlink_target(change.path[1])
                    try:
                        self.upload_symlink(change.path[1], target)
                    except errors.TransportNotPossible:
                        if not self.quiet:
                            self.outf.write('Not uploading symlink %s -> %s\n' % (change.path[1], target))
                else:
                    raise NotImplementedError
            for change in changes.modified:
                if self.is_ignored(change.path[1]):
                    if not self.quiet:
                        self.outf.write('Ignoring %s\n' % change.path[1])
                    continue
                if change.kind[1] == 'file':
                    self.upload_file(change.path[1], change.path[1])
                elif change.kind[1] == 'symlink':
                    target = self.tree.get_symlink_target(change.path[1])
                    self.upload_symlink(change.path[1], target)
                else:
                    raise NotImplementedError
            self.set_uploaded_revid(self.rev_id)