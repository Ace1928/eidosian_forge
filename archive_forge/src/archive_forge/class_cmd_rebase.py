from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
class cmd_rebase(Command):
    """Re-base a branch.

    Rebasing is the process of taking a branch and modifying the history so
    that it appears to start from a different point. This can be useful
    to clean up the history before submitting your changes. The tree at the
    end of the process will be the same as if you had merged the other branch,
    but the history will be different.

    The command takes the location of another branch on to which the branch in
    the specified directory (by default, the current working directory)
    will be rebased. If a branch is not specified then the parent branch
    is used, and this is usually the desired result.

    The first step identifies the revisions that are in the current branch that
    are not in the parent branch. The current branch is then set to be at the
    same revision as the target branch, and each revision is replayed on top
    of the branch. At the end of the process it will appear as though your
    current branch was branched off the current last revision of the target.

    Each revision that is replayed may cause conflicts in the tree. If this
    happens the command will stop and allow you to fix them up. Resolve the
    commits as you would for a merge, and then run 'brz resolve' to marked
    them as resolved. Once you have resolved all the conflicts you should
    run 'brz rebase-continue' to continue the rebase operation.

    If conflicts are encountered and you decide that you do not wish to continue
    you can run 'brz rebase-abort'.

    The '--onto' option allows you to specify a different revision in the
    target branch to start at when replaying the revisions. This means that
    you can change the point at which the current branch will appear to be
    branched from when the operation completes.
    """
    takes_args = ['upstream_location?']
    takes_options = ['revision', 'merge-type', 'verbose', Option('dry-run', help="Show what would be done, but don't actually do anything."), Option('always-rebase-merges', help="Don't skip revisions that merge already present revisions."), Option('pending-merges', help='Rebase pending merges onto local branch.'), Option('onto', help='Different revision to replay onto.', type=str), Option('directory', short_name='d', help='Branch to replay onto, rather than the one containing the working directory.', type=str)]

    @display_command
    def run(self, upstream_location=None, onto=None, revision=None, merge_type=None, verbose=False, dry_run=False, always_rebase_merges=False, pending_merges=False, directory='.'):
        from ...branch import Branch
        from ...revisionspec import RevisionSpec
        from ...workingtree import WorkingTree
        from .rebase import RebaseState1, WorkingTreeRevisionRewriter, generate_simple_plan, rebase, rebase_todo, regenerate_default_revid
        if revision is not None and pending_merges:
            raise CommandError(gettext('--revision and --pending-merges are mutually exclusive'))
        wt = WorkingTree.open_containing(directory)[0]
        wt.lock_write()
        try:
            state = RebaseState1(wt)
            if upstream_location is None:
                if pending_merges:
                    upstream_location = directory
                else:
                    upstream_location = wt.branch.get_parent()
                    if upstream_location is None:
                        raise CommandError(gettext('No upstream branch specified.'))
                    note(gettext('Rebasing on %s'), upstream_location)
            upstream = Branch.open_containing(upstream_location)[0]
            upstream_repository = upstream.repository
            upstream_revision = upstream.last_revision()
            if state.has_plan():
                raise CommandError(gettext("A rebase operation was interrupted. Continue using 'brz rebase-continue' or abort using 'brz rebase-abort'"))
            start_revid = None
            stop_revid = None
            if revision is not None:
                if len(revision) == 1:
                    if revision[0] is not None:
                        stop_revid = revision[0].as_revision_id(wt.branch)
                elif len(revision) == 2:
                    if revision[0] is not None:
                        start_revid = revision[0].as_revision_id(wt.branch)
                    if revision[1] is not None:
                        stop_revid = revision[1].as_revision_id(wt.branch)
                else:
                    raise CommandError(gettext('--revision takes only one or two arguments'))
            if pending_merges:
                wt_parents = wt.get_parent_ids()
                if len(wt_parents) in (0, 1):
                    raise CommandError(gettext('No pending merges present.'))
                elif len(wt_parents) > 2:
                    raise CommandError(gettext('Rebasing more than one pending merge not supported'))
                stop_revid = wt_parents[1]
                assert stop_revid is not None, 'stop revid invalid'
            if not pending_merges and wt.basis_tree().changes_from(wt).has_changed():
                raise UncommittedChanges(wt)
            wt.branch.repository.fetch(upstream_repository, upstream_revision)
            if onto is None:
                onto = upstream.last_revision()
            else:
                rev_spec = RevisionSpec.from_string(onto)
                onto = rev_spec.as_revision_id(upstream)
            wt.branch.repository.fetch(upstream_repository, revision_id=onto)
            if stop_revid is None:
                stop_revid = wt.branch.last_revision()
            repo_graph = wt.branch.repository.get_graph()
            our_new, onto_unique = repo_graph.find_difference(stop_revid, onto)
            if start_revid is None:
                if not onto_unique:
                    self.outf.write(gettext('No revisions to rebase.\n'))
                    return
                if not our_new:
                    self.outf.write(gettext('Base branch is descendant of current branch. Pulling instead.\n'))
                    if not dry_run:
                        wt.pull(upstream, stop_revision=onto)
                    return
            replace_map = generate_simple_plan(our_new, start_revid, stop_revid, onto, repo_graph, lambda revid, ps: regenerate_default_revid(wt.branch.repository, revid), not always_rebase_merges)
            if verbose or dry_run:
                todo = list(rebase_todo(wt.branch.repository, replace_map))
                note(gettext('%d revisions will be rebased:') % len(todo))
                for revid in todo:
                    note('%s' % revid)
            if not dry_run:
                state.write_plan(replace_map)
                replayer = WorkingTreeRevisionRewriter(wt, state, merge_type=merge_type)
                finish_rebase(state, wt, replace_map, replayer)
        finally:
            wt.unlock()