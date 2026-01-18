from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
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