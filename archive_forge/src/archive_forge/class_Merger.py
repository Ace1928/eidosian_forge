import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
class Merger:
    hooks = MergeHooks()
    merge_type: object

    def __init__(self, this_branch, other_tree=None, base_tree=None, this_tree=None, change_reporter=None, recurse='down', revision_graph=None):
        object.__init__(self)
        self.this_branch = this_branch
        self.this_basis = this_branch.last_revision()
        self.this_rev_id = None
        self.this_tree = this_tree
        self.this_revision_tree = None
        self.this_basis_tree = None
        self.other_tree = other_tree
        self.other_branch = None
        self.base_tree = base_tree
        self.ignore_zero = False
        self.backup_files = False
        self.interesting_files = None
        self.show_base = False
        self.reprocess = False
        self.pp = None
        self.recurse = recurse
        self.change_reporter = change_reporter
        self._cached_trees = {}
        self._revision_graph = revision_graph
        self._base_is_ancestor = None
        self._base_is_other_ancestor = None
        self._is_criss_cross = None
        self._lca_trees = None

    def cache_trees_with_revision_ids(self, trees):
        """Cache any tree in trees if it has a revision_id."""
        for maybe_tree in trees:
            if maybe_tree is None:
                continue
            try:
                rev_id = maybe_tree.get_revision_id()
            except AttributeError:
                continue
            self._cached_trees[rev_id] = maybe_tree

    @property
    def revision_graph(self):
        if self._revision_graph is None:
            self._revision_graph = self.this_branch.repository.get_graph()
        return self._revision_graph

    def _set_base_is_ancestor(self, value):
        self._base_is_ancestor = value

    def _get_base_is_ancestor(self):
        if self._base_is_ancestor is None:
            self._base_is_ancestor = self.revision_graph.is_ancestor(self.base_rev_id, self.this_basis)
        return self._base_is_ancestor
    base_is_ancestor = property(_get_base_is_ancestor, _set_base_is_ancestor)

    def _set_base_is_other_ancestor(self, value):
        self._base_is_other_ancestor = value

    def _get_base_is_other_ancestor(self):
        if self._base_is_other_ancestor is None:
            if self.other_basis is None:
                return True
            self._base_is_other_ancestor = self.revision_graph.is_ancestor(self.base_rev_id, self.other_basis)
        return self._base_is_other_ancestor
    base_is_other_ancestor = property(_get_base_is_other_ancestor, _set_base_is_other_ancestor)

    @staticmethod
    def from_uncommitted(tree, other_tree, base_tree=None):
        """Return a Merger for uncommitted changes in other_tree.

        :param tree: The tree to merge into
        :param other_tree: The tree to get uncommitted changes from
        :param base_tree: The basis to use for the merge.  If unspecified,
            other_tree.basis_tree() will be used.
        """
        if base_tree is None:
            base_tree = other_tree.basis_tree()
        merger = Merger(tree.branch, other_tree, base_tree, tree)
        merger.base_rev_id = merger.base_tree.get_revision_id()
        merger.other_rev_id = None
        merger.other_basis = merger.base_rev_id
        return merger

    @classmethod
    def from_mergeable(klass, tree, mergeable):
        """Return a Merger for a bundle or merge directive.

        :param tree: The tree to merge changes into
        :param mergeable: A merge directive or bundle
        """
        mergeable.install_revisions(tree.branch.repository)
        base_revision_id, other_revision_id, verified = mergeable.get_merge_request(tree.branch.repository)
        revision_graph = tree.branch.repository.get_graph()
        if base_revision_id is not None:
            if base_revision_id != _mod_revision.NULL_REVISION and revision_graph.is_ancestor(base_revision_id, tree.branch.last_revision()):
                base_revision_id = None
            else:
                trace.warning('Performing cherrypick')
        merger = klass.from_revision_ids(tree, other_revision_id, base_revision_id, revision_graph=revision_graph)
        return (merger, verified)

    @staticmethod
    def from_revision_ids(tree, other, base=None, other_branch=None, base_branch=None, revision_graph=None, tree_branch=None):
        """Return a Merger for revision-ids.

        :param tree: The tree to merge changes into
        :param other: The revision-id to use as OTHER
        :param base: The revision-id to use as BASE.  If not specified, will
            be auto-selected.
        :param other_branch: A branch containing the other revision-id.  If
            not supplied, tree.branch is used.
        :param base_branch: A branch containing the base revision-id.  If
            not supplied, other_branch or tree.branch will be used.
        :param revision_graph: If you have a revision_graph precomputed, pass
            it in, otherwise it will be created for you.
        :param tree_branch: The branch associated with tree.  If not supplied,
            tree.branch will be used.
        """
        if tree_branch is None:
            tree_branch = tree.branch
        merger = Merger(tree_branch, this_tree=tree, revision_graph=revision_graph)
        if other_branch is None:
            other_branch = tree.branch
        merger.set_other_revision(other, other_branch)
        if base is None:
            merger.find_base()
        else:
            if base_branch is None:
                base_branch = other_branch
            merger.set_base_revision(base, base_branch)
        return merger

    def revision_tree(self, revision_id, branch=None):
        if revision_id not in self._cached_trees:
            if branch is None:
                branch = self.this_branch
            try:
                tree = self.this_tree.revision_tree(revision_id)
            except errors.NoSuchRevisionInTree:
                tree = branch.repository.revision_tree(revision_id)
            self._cached_trees[revision_id] = tree
        return self._cached_trees[revision_id]

    def _get_tree(self, treespec, possible_transports=None):
        location, revno = treespec
        if revno is None:
            from .workingtree import WorkingTree
            tree = WorkingTree.open_containing(location)[0]
            return (tree.branch, tree)
        from .branch import Branch
        branch = Branch.open_containing(location, possible_transports)[0]
        if revno == -1:
            revision_id = branch.last_revision()
        else:
            revision_id = branch.get_rev_id(revno)
        return (branch, self.revision_tree(revision_id, branch))

    def set_interesting_files(self, file_list):
        self.interesting_files = file_list

    def set_pending(self):
        if not self.base_is_ancestor or not self.base_is_other_ancestor or self.other_rev_id is None:
            return
        self._add_parent()

    def _add_parent(self):
        new_parents = self.this_tree.get_parent_ids() + [self.other_rev_id]
        new_parent_trees = []
        with contextlib.ExitStack() as stack:
            for revision_id in new_parents:
                try:
                    tree = self.revision_tree(revision_id)
                except errors.NoSuchRevision:
                    tree = None
                else:
                    stack.enter_context(tree.lock_read())
                new_parent_trees.append((revision_id, tree))
            self.this_tree.set_parent_trees(new_parent_trees, allow_leftmost_as_ghost=True)

    def set_other(self, other_revision, possible_transports=None):
        """Set the revision and tree to merge from.

        This sets the other_tree, other_rev_id, other_basis attributes.

        :param other_revision: The [path, revision] list to merge from.
        """
        self.other_branch, self.other_tree = self._get_tree(other_revision, possible_transports)
        if other_revision[1] == -1:
            self.other_rev_id = self.other_branch.last_revision()
            if _mod_revision.is_null(self.other_rev_id):
                raise errors.NoCommits(self.other_branch)
            self.other_basis = self.other_rev_id
        elif other_revision[1] is not None:
            self.other_rev_id = self.other_branch.get_rev_id(other_revision[1])
            self.other_basis = self.other_rev_id
        else:
            self.other_rev_id = None
            self.other_basis = self.other_branch.last_revision()
            if self.other_basis is None:
                raise errors.NoCommits(self.other_branch)
        if self.other_rev_id is not None:
            self._cached_trees[self.other_rev_id] = self.other_tree
        self._maybe_fetch(self.other_branch, self.this_branch, self.other_basis)

    def set_other_revision(self, revision_id, other_branch):
        """Set 'other' based on a branch and revision id

        :param revision_id: The revision to use for a tree
        :param other_branch: The branch containing this tree
        """
        self.other_rev_id = revision_id
        self.other_branch = other_branch
        self._maybe_fetch(other_branch, self.this_branch, self.other_rev_id)
        self.other_tree = self.revision_tree(revision_id)
        self.other_basis = revision_id

    def set_base_revision(self, revision_id, branch):
        """Set 'base' based on a branch and revision id

        :param revision_id: The revision to use for a tree
        :param branch: The branch containing this tree
        """
        self.base_rev_id = revision_id
        self.base_branch = branch
        self._maybe_fetch(branch, self.this_branch, revision_id)
        self.base_tree = self.revision_tree(revision_id)

    def _maybe_fetch(self, source, target, revision_id):
        if not source.repository.has_same_location(target.repository):
            target.fetch(source, revision_id)

    def find_base(self):
        revisions = [self.this_basis, self.other_basis]
        if _mod_revision.NULL_REVISION in revisions:
            self.base_rev_id = _mod_revision.NULL_REVISION
            self.base_tree = self.revision_tree(self.base_rev_id)
            self._is_criss_cross = False
        else:
            lcas = self.revision_graph.find_lca(revisions[0], revisions[1])
            self._is_criss_cross = False
            if len(lcas) == 0:
                self.base_rev_id = _mod_revision.NULL_REVISION
            elif len(lcas) == 1:
                self.base_rev_id = list(lcas)[0]
            else:
                self._is_criss_cross = True
                if len(lcas) > 2:
                    self.base_rev_id = self.revision_graph.find_unique_lca(revisions[0], revisions[1])
                else:
                    self.base_rev_id = self.revision_graph.find_unique_lca(*lcas)
                sorted_lca_keys = self.revision_graph.find_merge_order(revisions[0], lcas)
                if self.base_rev_id == _mod_revision.NULL_REVISION:
                    self.base_rev_id = sorted_lca_keys[0]
            if self.base_rev_id == _mod_revision.NULL_REVISION:
                raise errors.UnrelatedBranches()
            if self._is_criss_cross:
                trace.warning('Warning: criss-cross merge encountered.  See bzr help criss-cross.')
                trace.mutter('Criss-cross lcas: %r' % lcas)
                if self.base_rev_id in lcas:
                    trace.mutter('Unable to find unique lca. Fallback %r as best option.' % self.base_rev_id)
                interesting_revision_ids = set(lcas)
                interesting_revision_ids.add(self.base_rev_id)
                interesting_trees = {t.get_revision_id(): t for t in self.this_branch.repository.revision_trees(interesting_revision_ids)}
                self._cached_trees.update(interesting_trees)
                if self.base_rev_id in lcas:
                    self.base_tree = interesting_trees[self.base_rev_id]
                else:
                    self.base_tree = interesting_trees.pop(self.base_rev_id)
                self._lca_trees = [interesting_trees[key] for key in sorted_lca_keys]
            else:
                self.base_tree = self.revision_tree(self.base_rev_id)
        self.base_is_ancestor = True
        self.base_is_other_ancestor = True
        trace.mutter('Base revid: %r' % self.base_rev_id)

    def set_base(self, base_revision):
        """Set the base revision to use for the merge.

        :param base_revision: A 2-list containing a path and revision number.
        """
        trace.mutter('doing merge() with no base_revision specified')
        if base_revision == [None, None]:
            self.find_base()
        else:
            base_branch, self.base_tree = self._get_tree(base_revision)
            if base_revision[1] == -1:
                self.base_rev_id = base_branch.last_revision()
            elif base_revision[1] is None:
                self.base_rev_id = _mod_revision.NULL_REVISION
            else:
                self.base_rev_id = base_branch.get_rev_id(base_revision[1])
            self._maybe_fetch(base_branch, self.this_branch, self.base_rev_id)

    def make_merger(self):
        kwargs = {'working_tree': self.this_tree, 'this_tree': self.this_tree, 'other_tree': self.other_tree, 'interesting_files': self.interesting_files, 'this_branch': self.this_branch, 'other_branch': self.other_branch, 'do_merge': False}
        if self.merge_type.requires_base:
            kwargs['base_tree'] = self.base_tree
        if self.merge_type.supports_reprocess:
            kwargs['reprocess'] = self.reprocess
        elif self.reprocess:
            raise errors.BzrError('Conflict reduction is not supported for merge type %s.' % self.merge_type)
        if self.merge_type.supports_show_base:
            kwargs['show_base'] = self.show_base
        elif self.show_base:
            raise errors.BzrError('Showing base is not supported for this merge type. %s' % self.merge_type)
        if not getattr(self.merge_type, 'supports_reverse_cherrypick', True) and (not self.base_is_other_ancestor):
            raise errors.CannotReverseCherrypick()
        if self.merge_type.supports_cherrypick:
            kwargs['cherrypick'] = not self.base_is_ancestor or not self.base_is_other_ancestor
        if self._is_criss_cross and getattr(self.merge_type, 'supports_lca_trees', False):
            kwargs['lca_trees'] = self._lca_trees
        return self.merge_type(change_reporter=self.change_reporter, **kwargs)

    def _do_merge_to(self):
        merge = self.make_merger()
        if self.other_branch is not None:
            self.other_branch.update_references(self.this_branch)
        for hook in Merger.hooks['pre_merge']:
            hook(merge)
        merge.do_merge()
        for hook in Merger.hooks['post_merge']:
            hook(merge)
        if self.recurse == 'down':
            for relpath in self.this_tree.iter_references():
                sub_tree = self.this_tree.get_nested_tree(relpath)
                other_revision = self.other_tree.get_reference_revision(relpath)
                if other_revision == sub_tree.last_revision():
                    continue
                other_branch = self.other_tree.reference_parent(relpath)
                graph = self.this_tree.branch.repository.get_graph(other_branch.repository)
                if graph.is_ancestor(sub_tree.last_revision(), other_revision):
                    sub_tree.pull(other_branch, stop_revision=other_revision)
                else:
                    sub_merge = Merger(sub_tree.branch, this_tree=sub_tree)
                    sub_merge.merge_type = self.merge_type
                    sub_merge.set_other_revision(other_revision, other_branch)
                    base_tree_path = _mod_tree.find_previous_path(self.this_tree, self.base_tree, relpath)
                    if base_tree_path is None:
                        raise NotImplementedError
                    base_revision = self.base_tree.get_reference_revision(base_tree_path)
                    sub_merge.base_tree = sub_tree.branch.repository.revision_tree(base_revision)
                    sub_merge.base_rev_id = base_revision
                    sub_merge.do_merge()
        return merge

    def do_merge(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.this_tree.lock_tree_write())
            if self.base_tree is not None:
                stack.enter_context(self.base_tree.lock_read())
            if self.other_tree is not None:
                stack.enter_context(self.other_tree.lock_read())
            merge = self._do_merge_to()
        if len(merge.cooked_conflicts) == 0:
            if not self.ignore_zero and (not trace.is_quiet()):
                trace.note(gettext('All changes applied successfully.'))
        else:
            trace.note(gettext('%d conflicts encountered.') % len(merge.cooked_conflicts))
        return merge.cooked_conflicts