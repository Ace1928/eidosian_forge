from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
class cmd_upload(commands.Command):
    """Upload a working tree, as a whole or incrementally.

    If no destination is specified use the last one used.
    If no revision is specified upload the changes since the last upload.

    Changes include files added, renamed, modified or removed.
    """
    _see_also = ['plugins/upload']
    takes_args = ['location?']
    takes_options = ['revision', 'remember', 'overwrite', option.Option('full', 'Upload the full working tree.'), option.Option('quiet', 'Do not output what is being done.', short_name='q'), option.Option('directory', help='Branch to upload from, rather than the one containing the working directory.', short_name='d', type=str), option.Option('auto', 'Trigger an upload from this branch whenever the tip revision changes.')]

    def run(self, location=None, full=False, revision=None, remember=None, directory=None, quiet=False, auto=None, overwrite=False):
        if directory is None:
            directory = '.'
        wt, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch(directory)
        if wt:
            locked = wt
        else:
            locked = branch
        with locked.lock_read():
            if wt:
                changes = wt.changes_from(wt.basis_tree())
                if revision is None and changes.has_changed():
                    raise errors.UncommittedChanges(wt)
            conf = branch.get_config_stack()
            if location is None:
                stored_loc = conf.get('upload_location')
                if stored_loc is None:
                    raise errors.CommandError('No upload location known or specified.')
                else:
                    display_url = urlutils.unescape_for_display(stored_loc, self.outf.encoding)
                    self.outf.write('Using saved location: %s\n' % display_url)
                    location = stored_loc
            to_transport = transport.get_transport(location)
            try:
                to_bzr_dir = controldir.ControlDir.open_from_transport(to_transport)
                has_wt = to_bzr_dir.has_workingtree()
            except errors.NotBranchError:
                has_wt = False
            except errors.NotLocalUrl:
                has_wt = True
            if has_wt:
                raise CannotUploadToWorkingTree(url=location)
            if revision is None:
                rev_id = branch.last_revision()
            else:
                if len(revision) != 1:
                    raise errors.CommandError('bzr upload --revision takes exactly 1 argument')
                rev_id = revision[0].in_history(branch).rev_id
            tree = branch.repository.revision_tree(rev_id)
            uploader = BzrUploader(branch, to_transport, self.outf, tree, rev_id, quiet=quiet)
            if not overwrite:
                prev_uploaded_rev_id = uploader.get_uploaded_revid()
                graph = branch.repository.get_graph()
                if not graph.is_ancestor(prev_uploaded_rev_id, rev_id):
                    raise DivergedUploadedTree(revid=rev_id, uploaded_revid=prev_uploaded_rev_id)
            if full:
                uploader.upload_full_tree()
            else:
                uploader.upload_tree()
        with branch.lock_write():
            upload_location = conf.get('upload_location')
            if upload_location is None or remember:
                conf.set('upload_location', urlutils.unescape(to_transport.base))
            if auto is not None:
                conf.set('upload_auto', auto)