import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_merge(Command):
    __doc__ = 'Perform a three-way merge.\n\n    The source of the merge can be specified either in the form of a branch,\n    or in the form of a path to a file containing a merge directive generated\n    with brz send. If neither is specified, the default is the upstream branch\n    or the branch most recently merged using --remember.  The source of the\n    merge may also be specified in the form of a path to a file in another\n    branch:  in this case, only the modifications to that file are merged into\n    the current working tree.\n\n    When merging from a branch, by default brz will try to merge in all new\n    work from the other branch, automatically determining an appropriate base\n    revision.  If this fails, you may need to give an explicit base.\n\n    To pick a different ending revision, pass "--revision OTHER".  brz will\n    try to merge in all new work up to and including revision OTHER.\n\n    If you specify two values, "--revision BASE..OTHER", only revisions BASE\n    through OTHER, excluding BASE but including OTHER, will be merged.  If this\n    causes some revisions to be skipped, i.e. if the destination branch does\n    not already contain revision BASE, such a merge is commonly referred to as\n    a "cherrypick". Unlike a normal merge, Breezy does not currently track\n    cherrypicks. The changes look like a normal commit, and the history of the\n    changes from the other branch is not stored in the commit.\n\n    Revision numbers are always relative to the source branch.\n\n    Merge will do its best to combine the changes in two branches, but there\n    are some kinds of problems only a human can fix.  When it encounters those,\n    it will mark a conflict.  A conflict means that you need to fix something,\n    before you can commit.\n\n    Use brz resolve when you have fixed a problem.  See also brz conflicts.\n\n    If there is no default branch set, the first merge will set it (use\n    --no-remember to avoid setting it). After that, you can omit the branch\n    to use the default.  To change the default, use --remember. The value will\n    only be saved if the remote location can be accessed.\n\n    The results of the merge are placed into the destination working\n    directory, where they can be reviewed (with brz diff), tested, and then\n    committed to record the result of the merge.\n\n    merge refuses to run if there are any uncommitted changes, unless\n    --force is given.  If --force is given, then the changes from the source\n    will be merged with the current working tree, including any uncommitted\n    changes in the tree.  The --force option can also be used to create a\n    merge revision which has more than two parents.\n\n    If one would like to merge changes from the working tree of the other\n    branch without merging any committed revisions, the --uncommitted option\n    can be given.\n\n    To select only some changes to merge, use "merge -i", which will prompt\n    you to apply each diff hunk and file change, similar to "shelve".\n\n    :Examples:\n        To merge all new revisions from brz.dev::\n\n            brz merge ../brz.dev\n\n        To merge changes up to and including revision 82 from brz.dev::\n\n            brz merge -r 82 ../brz.dev\n\n        To merge the changes introduced by 82, without previous changes::\n\n            brz merge -r 81..82 ../brz.dev\n\n        To apply a merge directive contained in /tmp/merge::\n\n            brz merge /tmp/merge\n\n        To create a merge revision with three parents from two branches\n        feature1a and feature1b:\n\n            brz merge ../feature1a\n            brz merge ../feature1b --force\n            brz commit -m \'revision with three parents\'\n    '
    encoding_type = 'exact'
    _see_also = ['update', 'remerge', 'status-flags', 'send']
    takes_args = ['location?']
    takes_options = ['change', 'revision', Option('force', help='Merge even if the destination tree has uncommitted changes.'), 'merge-type', 'reprocess', 'remember', Option('show-base', help='Show base revision text in conflicts.'), Option('uncommitted', help='Apply uncommitted changes from a working copy, instead of branch changes.'), Option('pull', help='If the destination is already completely merged into the source, pull from the source rather than merging.  When this happens, you do not need to commit the result.'), custom_help('directory', help='Branch to merge into, rather than the one containing the working directory.'), Option('preview', help='Instead of merging, show a diff of the merge.'), Option('interactive', help='Select changes interactively.', short_name='i')]

    def run(self, location=None, revision=None, force=False, merge_type=None, show_base=False, reprocess=None, remember=None, uncommitted=False, pull=False, directory=None, preview=False, interactive=False):
        if merge_type is None:
            merge_type = _mod_merge.Merge3Merger
        if directory is None:
            directory = '.'
        possible_transports = []
        merger = None
        allow_pending = True
        verified = 'inapplicable'
        tree = WorkingTree.open_containing(directory)[0]
        if tree.branch.last_revision() == _mod_revision.NULL_REVISION:
            raise errors.CommandError(gettext('Merging into empty branches not currently supported, https://bugs.launchpad.net/bzr/+bug/308562'))
        if not force:
            if tree.has_changes():
                raise errors.UncommittedChanges(tree)
        view_info = _get_view_info_for_change_reporter(tree)
        change_reporter = delta._ChangeReporter(unversioned_filter=tree.is_ignored, view_info=view_info)
        pb = ui.ui_factory.nested_progress_bar()
        self.enter_context(pb)
        self.enter_context(tree.lock_write())
        if location is not None:
            try:
                mergeable = _mod_mergeable.read_mergeable_from_url(location, possible_transports=possible_transports)
            except errors.NotABundle:
                mergeable = None
            else:
                if uncommitted:
                    raise errors.CommandError(gettext('Cannot use --uncommitted with bundles or merge directives.'))
                if revision is not None:
                    raise errors.CommandError(gettext('Cannot use -r with merge directives or bundles'))
                merger, verified = _mod_merge.Merger.from_mergeable(tree, mergeable)
        if merger is None and uncommitted:
            if revision is not None and len(revision) > 0:
                raise errors.CommandError(gettext('Cannot use --uncommitted and --revision at the same time.'))
            merger = self.get_merger_from_uncommitted(tree, location, None)
            allow_pending = False
        if merger is None:
            merger, allow_pending = self._get_merger_from_branch(tree, location, revision, remember, possible_transports, None)
        merger.merge_type = merge_type
        merger.reprocess = reprocess
        merger.show_base = show_base
        self.sanity_check_merger(merger)
        if merger.base_rev_id == merger.other_rev_id and merger.other_rev_id is not None:
            if merger.interesting_files:
                if not merger.other_tree.has_filename(merger.interesting_files[0]):
                    note(gettext('merger: ') + str(merger))
                    raise errors.PathsDoNotExist([location])
            note(gettext('Nothing to do.'))
            return 0
        if pull and (not preview):
            if merger.interesting_files is not None:
                raise errors.CommandError(gettext('Cannot pull individual files'))
            if merger.base_rev_id == tree.last_revision():
                result = tree.pull(merger.other_branch, False, merger.other_rev_id)
                result.report(self.outf)
                return 0
        if merger.this_basis is None:
            raise errors.CommandError(gettext("This branch has no commits. (perhaps you would prefer 'brz pull')"))
        if preview:
            return self._do_preview(merger)
        elif interactive:
            return self._do_interactive(merger)
        else:
            return self._do_merge(merger, change_reporter, allow_pending, verified)

    def _get_preview(self, merger):
        tree_merger = merger.make_merger()
        tt = tree_merger.make_preview_transform()
        self.enter_context(tt)
        result_tree = tt.get_preview_tree()
        return result_tree

    def _do_preview(self, merger):
        from .diff import show_diff_trees
        result_tree = self._get_preview(merger)
        path_encoding = osutils.get_diff_header_encoding()
        show_diff_trees(merger.this_tree, result_tree, self.outf, old_label='', new_label='', path_encoding=path_encoding)

    def _do_merge(self, merger, change_reporter, allow_pending, verified):
        merger.change_reporter = change_reporter
        conflict_count = len(merger.do_merge())
        if allow_pending:
            merger.set_pending()
        if verified == 'failed':
            warning('Preview patch does not match changes')
        if conflict_count != 0:
            return 1
        else:
            return 0

    def _do_interactive(self, merger):
        """Perform an interactive merge.

        This works by generating a preview tree of the merge, then using
        Shelver to selectively remove the differences between the working tree
        and the preview tree.
        """
        from . import shelf_ui
        result_tree = self._get_preview(merger)
        writer = breezy.option.diff_writer_registry.get()
        shelver = shelf_ui.Shelver(merger.this_tree, result_tree, destroy=True, reporter=shelf_ui.ApplyReporter(), diff_writer=writer(self.outf))
        try:
            shelver.run()
        finally:
            shelver.finalize()

    def sanity_check_merger(self, merger):
        if merger.show_base and merger.merge_type is not _mod_merge.Merge3Merger:
            raise errors.CommandError(gettext('Show-base is not supported for this merge type. %s') % merger.merge_type)
        if merger.reprocess is None:
            if merger.show_base:
                merger.reprocess = False
            else:
                merger.reprocess = merger.merge_type.supports_reprocess
        if merger.reprocess and (not merger.merge_type.supports_reprocess):
            raise errors.CommandError(gettext('Conflict reduction is not supported for merge type %s.') % merger.merge_type)
        if merger.reprocess and merger.show_base:
            raise errors.CommandError(gettext('Cannot do conflict reduction and show base.'))
        if merger.merge_type.requires_file_merge_plan and (not getattr(merger.this_tree, 'plan_file_merge', None) or not getattr(merger.other_tree, 'plan_file_merge', None) or (merger.base_tree is not None and (not getattr(merger.base_tree, 'plan_file_merge', None)))):
            raise errors.CommandError(gettext('Plan file merge unsupported: Merge type incompatible with tree formats.'))

    def _get_merger_from_branch(self, tree, location, revision, remember, possible_transports, pb):
        """Produce a merger from a location, assuming it refers to a branch."""
        other_loc, user_location = self._select_branch_location(tree, location, revision, -1)
        if revision is not None and len(revision) == 2:
            base_loc, _unused = self._select_branch_location(tree, location, revision, 0)
        else:
            base_loc = other_loc
        other_branch, other_path = Branch.open_containing(other_loc, possible_transports)
        if base_loc == other_loc:
            base_branch = other_branch
        else:
            base_branch, base_path = Branch.open_containing(base_loc, possible_transports)
        other_revision_id = None
        base_revision_id = None
        if revision is not None:
            if len(revision) >= 1:
                other_revision_id = revision[-1].as_revision_id(other_branch)
            if len(revision) == 2:
                base_revision_id = revision[0].as_revision_id(base_branch)
        if other_revision_id is None:
            other_revision_id = other_branch.last_revision()
        if user_location is not None and (remember or (remember is None and tree.branch.get_submit_branch() is None)):
            tree.branch.set_submit_branch(other_branch.base)
        other_branch.tags.merge_to(tree.branch.tags, ignore_master=True)
        merger = _mod_merge.Merger.from_revision_ids(tree, other_revision_id, base_revision_id, other_branch, base_branch)
        if other_path != '':
            allow_pending = False
            merger.interesting_files = [other_path]
        else:
            allow_pending = True
        return (merger, allow_pending)

    def get_merger_from_uncommitted(self, tree, location, pb):
        """Get a merger for uncommitted changes.

        :param tree: The tree the merger should apply to.
        :param location: The location containing uncommitted changes.
        :param pb: The progress bar to use for showing progress.
        """
        location = self._select_branch_location(tree, location)[0]
        other_tree, other_path = WorkingTree.open_containing(location)
        merger = _mod_merge.Merger.from_uncommitted(tree, other_tree, pb)
        if other_path != '':
            merger.interesting_files = [other_path]
        return merger

    def _select_branch_location(self, tree, user_location, revision=None, index=None):
        """Select a branch location, according to possible inputs.

        If provided, branches from ``revision`` are preferred.  (Both
        ``revision`` and ``index`` must be supplied.)

        Otherwise, the ``location`` parameter is used.  If it is None, then the
        ``submit`` or ``parent`` location is used, and a note is printed.

        :param tree: The working tree to select a branch for merging into
        :param location: The location entered by the user
        :param revision: The revision parameter to the command
        :param index: The index to use for the revision parameter.  Negative
            indices are permitted.
        :return: (selected_location, user_location).  The default location
            will be the user-entered location.
        """
        if revision is not None and index is not None and (revision[index] is not None):
            branch = revision[index].get_branch()
            if branch is not None:
                return (branch, branch)
        if user_location is None:
            location = self._get_remembered(tree, 'Merging from')
        else:
            location = user_location
        return (location, user_location)

    def _get_remembered(self, tree, verb_string):
        """Use tree.branch's parent if none was supplied.

        Report if the remembered location was used.
        """
        stored_location = tree.branch.get_submit_branch()
        stored_location_type = 'submit'
        if stored_location is None:
            stored_location = tree.branch.get_parent()
            stored_location_type = 'parent'
        mutter('%s', stored_location)
        if stored_location is None:
            raise errors.CommandError(gettext('No location specified or remembered'))
        display_url = urlutils.unescape_for_display(stored_location, 'utf-8')
        note(gettext('{0} remembered {1} location {2}').format(verb_string, stored_location_type, display_url))
        return stored_location