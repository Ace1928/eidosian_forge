from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
class cmd_rebase_foreign(Command):
    """Rebase revisions based on a branch created with a different import tool.

    This will change the identity of revisions whose parents
    were mapped from revisions in the other version control system.

    You are recommended to run "brz check" in the local repository
    after running this command.
    """
    takes_args = ['new_base?']
    takes_options = ['verbose', Option('idmap-file', help='Write map with old and new revision ids.', type=str), Option('directory', short_name='d', help='Branch to replay onto, rather than the one containing the working directory.', type=str)]

    def run(self, new_base=None, verbose=False, idmap_file=None, directory='.'):
        from ... import urlutils
        from ...branch import Branch
        from ...foreign import update_workingtree_fileids
        from ...workingtree import WorkingTree
        from .pseudonyms import find_pseudonyms, generate_rebase_map_from_pseudonyms, pseudonyms_as_dict
        from .upgrade import create_deterministic_revid, upgrade_branch
        try:
            wt_to = WorkingTree.open(directory)
            branch_to = wt_to.branch
        except NoWorkingTree:
            wt_to = None
            branch_to = Branch.open(directory)
        stored_loc = branch_to.get_parent()
        if new_base is None:
            if stored_loc is None:
                raise CommandError(gettext('No pull location known or specified.'))
            else:
                display_url = urlutils.unescape_for_display(stored_loc, self.outf.encoding)
                self.outf.write(gettext('Using saved location: %s\n') % display_url)
                new_base = Branch.open(stored_loc)
        else:
            new_base = Branch.open(new_base)
        branch_to.repository.fetch(new_base.repository, revision_id=branch_to.last_revision())
        pseudonyms = pseudonyms_as_dict(find_pseudonyms(branch_to.repository, branch_to.repository.all_revision_ids()))

        def generate_rebase_map(revision_id):
            return generate_rebase_map_from_pseudonyms(pseudonyms, branch_to.repository.get_ancestry(revision_id), branch_to.repository.get_ancestry(new_base.last_revision()))

        def determine_new_revid(old_revid, new_parents):
            return create_deterministic_revid(old_revid, new_parents)
        branch_to.lock_write()
        try:
            graph = branch_to.repository.get_graph()
            renames = upgrade_branch(branch_to, generate_rebase_map, determine_new_revid, allow_changes=True, verbose=verbose)
            if wt_to is not None:
                basis_tree = wt_to.basis_tree()
                basis_tree.lock_read()
                try:
                    update_workingtree_fileids(wt_to, basis_tree)
                finally:
                    basis_tree.unlock()
        finally:
            branch_to.unlock()
        if renames == {}:
            note(gettext('Nothing to do.'))
        if idmap_file is not None:
            f = open(idmap_file, 'w')
            try:
                for oldid, newid in renames.iteritems():
                    f.write('{}\t{}\n'.format(oldid, newid))
            finally:
                f.close()
        if wt_to is not None:
            wt_to.set_last_revision(branch_to.last_revision())