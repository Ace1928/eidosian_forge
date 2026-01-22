from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
class VersionedFileRepoReconciler:
    """Reconciler that reconciles a repository.

    The goal of repository reconciliation is to make any derived data
    consistent with the core data committed by a user. This can involve
    reindexing, or removing unreferenced data if that can interfere with
    queries in a given repository.

    Currently this consists of an inventory reweave with revision cross-checks.
    """

    def __init__(self, repo, other=None, thorough=False):
        """Construct a RepoReconciler.

        :param thorough: perform a thorough check which may take longer but
                         will correct non-data loss issues such as incorrect
                         cached data.
        """
        self.garbage_inventories = 0
        self.inconsistent_parents = 0
        self.aborted = False
        self.repo = repo
        self.thorough = thorough

    def reconcile(self):
        """Perform reconciliation.

        After reconciliation the following attributes document found issues:

        * `inconsistent_parents`: The number of revisions in the repository
          whose ancestry was being reported incorrectly.
        * `garbage_inventories`: The number of inventory objects without
          revisions that were garbage collected.
        """
        with self.repo.lock_write(), ui.ui_factory.nested_progress_bar() as self.pb:
            self._reconcile_steps()
            ret = ReconcileResult()
            ret.aborted = self.aborted
            ret.garbage_inventories = self.garbage_inventories
            ret.inconsistent_parents = self.inconsistent_parents
            return ret

    def _reconcile_steps(self):
        """Perform the steps to reconcile this repository."""
        self._reweave_inventory()

    def _reweave_inventory(self):
        """Regenerate the inventory weave for the repository from scratch.

        This is a smart function: it will only do the reweave if doing it
        will correct data issues. The self.thorough flag controls whether
        only data-loss causing issues (!self.thorough) or all issues
        (self.thorough) are treated as requiring the reweave.
        """
        transaction = self.repo.get_transaction()
        self.pb.update(gettext('Reading inventory data'))
        self.inventory = self.repo.inventories
        self.revisions = self.repo.revisions
        self.pending = {key[-1] for key in self.revisions.keys()}
        self._rev_graph = {}
        self.inconsistent_parents = 0
        self._setup_steps(len(self.pending))
        for rev_id in self.pending:
            self._graph_revision(rev_id)
        self._check_garbage_inventories()
        if not self.inconsistent_parents and (not self.garbage_inventories or not self.thorough):
            ui.ui_factory.note(gettext('Inventory ok.'))
            return
        self.pb.update(gettext('Backing up inventory'), 0, 0)
        self.repo._backup_inventory()
        ui.ui_factory.note(gettext('Backup inventory created.'))
        new_inventories = self.repo._temp_inventories()
        self._setup_steps(len(self._rev_graph))
        revision_keys = [(rev_id,) for rev_id in topo_sort(self._rev_graph)]
        stream = self._change_inv_parents(self.inventory.get_record_stream(revision_keys, 'unordered', True), self._new_inv_parents, set(revision_keys))
        new_inventories.insert_record_stream(stream)
        if not set(new_inventories.keys()) == {(revid,) for revid in self.pending}:
            raise AssertionError()
        self.pb.update(gettext('Writing weave'))
        self.repo._activate_new_inventory()
        self.inventory = None
        ui.ui_factory.note(gettext('Inventory regenerated.'))

    def _new_inv_parents(self, revision_key):
        """Lookup ghost-filtered parents for revision_key."""
        return tuple([(revid,) for revid in self._rev_graph[revision_key[-1]]])

    def _change_inv_parents(self, stream, get_parents, all_revision_keys):
        """Adapt a record stream to reconcile the parents."""
        for record in stream:
            wanted_parents = get_parents(record.key)
            if wanted_parents and wanted_parents[0] not in all_revision_keys:
                chunks = record.get_bytes_as('chunked')
                yield ChunkedContentFactory(record.key, wanted_parents, record.sha1, chunks)
            else:
                adapted_record = AdapterFactory(record.key, wanted_parents, record)
                yield adapted_record
            self._reweave_step('adding inventories')

    def _setup_steps(self, new_total):
        """Setup the markers we need to control the progress bar."""
        self.total = new_total
        self.count = 0

    def _graph_revision(self, rev_id):
        """Load a revision into the revision graph."""
        self._reweave_step('loading revisions')
        rev = self.repo.get_revision_reconcile(rev_id)
        parents = []
        for parent in rev.parent_ids:
            if self._parent_is_available(parent):
                parents.append(parent)
            else:
                mutter('found ghost %s', parent)
        self._rev_graph[rev_id] = parents

    def _check_garbage_inventories(self):
        """Check for garbage inventories which we cannot trust

        We cant trust them because their pre-requisite file data may not
        be present - all we know is that their revision was not installed.
        """
        if not self.thorough:
            return
        inventories = set(self.inventory.keys())
        revisions = set(self.revisions.keys())
        garbage = inventories.difference(revisions)
        self.garbage_inventories = len(garbage)
        for revision_key in garbage:
            mutter('Garbage inventory {%s} found.', revision_key[-1])

    def _parent_is_available(self, parent):
        """True if parent is a fully available revision

        A fully available revision has a inventory and a revision object in the
        repository.
        """
        if parent in self._rev_graph:
            return True
        inv_present = 1 == len(self.inventory.get_parent_map([(parent,)]))
        return inv_present and self.repo.has_revision(parent)

    def _reweave_step(self, message):
        """Mark a single step of regeneration complete."""
        self.pb.update(message, self.count, self.total)
        self.count += 1