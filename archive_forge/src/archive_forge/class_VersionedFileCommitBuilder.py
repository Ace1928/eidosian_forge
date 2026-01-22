from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
class VersionedFileCommitBuilder(CommitBuilder):
    """Commit builder implementation for versioned files based repositories.
    """

    def __init__(self, repository, parents, config_stack, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False, owns_transaction=True):
        super().__init__(repository, parents, config_stack, timestamp, timezone, committer, revprops, revision_id, lossy)
        try:
            basis_id = self.parents[0]
        except IndexError:
            basis_id = _mod_revision.NULL_REVISION
        self.basis_delta_revision = basis_id
        self._new_inventory = None
        self._basis_delta = []
        self.__heads = graph.HeadsCache(repository.get_graph()).heads
        self._any_changes = False
        self._owns_transaction = owns_transaction

    def any_changes(self):
        """Return True if any entries were changed.

        This includes merge-only changes. It is the core for the --unchanged
        detection in commit.

        :return: True if any changes have occured.
        """
        return self._any_changes

    def _ensure_fallback_inventories(self):
        """Ensure that appropriate inventories are available.

        This only applies to repositories that are stacked, and is about
        enusring the stacking invariants. Namely, that for any revision that is
        present, we either have all of the file content, or we have the parent
        inventory and the delta file content.
        """
        if not self.repository._fallback_repositories:
            return
        if not self.repository._format.supports_chks:
            raise errors.BzrError('Cannot commit directly to a stacked branch in pre-2a formats. See https://bugs.launchpad.net/bzr/+bug/375013 for details.')
        parent_keys = [(p,) for p in self.parents]
        parent_map = self.repository.inventories._index.get_parent_map(parent_keys)
        missing_parent_keys = {pk for pk in parent_keys if pk not in parent_map}
        fallback_repos = list(reversed(self.repository._fallback_repositories))
        missing_keys = [('inventories', pk[0]) for pk in missing_parent_keys]
        resume_tokens = []
        while missing_keys and fallback_repos:
            fallback_repo = fallback_repos.pop()
            source = fallback_repo._get_source(self.repository._format)
            sink = self.repository._get_sink()
            missing_keys = sink.insert_missing_keys(source, missing_keys)
        if missing_keys:
            raise errors.BzrError('Unable to fill in parent inventories for a stacked branch')

    def commit(self, message):
        """Make the actual commit.

        :return: The revision id of the recorded revision.
        """
        self._validate_unicode_text(message, 'commit message')
        rev = _mod_revision.Revision(timestamp=self._timestamp, timezone=self._timezone, committer=self._committer, message=message, inventory_sha1=self.inv_sha1, revision_id=self._new_revision_id, properties=self._revprops)
        rev.parent_ids = self.parents
        create_signatures = self._config_stack.get('create_signatures')
        if create_signatures in (_mod_config.SIGN_ALWAYS, _mod_config.SIGN_WHEN_POSSIBLE):
            testament = Testament(rev, self.revision_tree())
            plaintext = testament.as_short_text()
            try:
                self.repository.store_revision_signature(gpg.GPGStrategy(self._config_stack), plaintext, self._new_revision_id)
            except gpg.GpgNotInstalled as e:
                if create_signatures == _mod_config.SIGN_WHEN_POSSIBLE:
                    note('skipping commit signature: %s', e)
                else:
                    raise
            except gpg.SigningFailed as e:
                if create_signatures == _mod_config.SIGN_WHEN_POSSIBLE:
                    note('commit signature failed: %s', e)
                else:
                    raise
        self.repository._add_revision(rev)
        self._ensure_fallback_inventories()
        if self._owns_transaction:
            self.repository.commit_write_group()
        return self._new_revision_id

    def abort(self):
        """Abort the commit that is being built.
        """
        if self._owns_transaction:
            self.repository.abort_write_group()

    def revision_tree(self):
        """Return the tree that was just committed.

        After calling commit() this can be called to get a
        RevisionTree representing the newly committed tree. This is
        preferred to calling Repository.revision_tree() because that may
        require deserializing the inventory, while we already have a copy in
        memory.
        """
        if self._new_inventory is None:
            self._new_inventory = self.repository.get_inventory(self._new_revision_id)
        return inventorytree.InventoryRevisionTree(self.repository, self._new_inventory, self._new_revision_id)

    def finish_inventory(self):
        """Tell the builder that the inventory is finished.

        :return: The inventory id in the repository, which can be used with
            repository.get_inventory.
        """
        basis_id = self.basis_delta_revision
        self.inv_sha1, self._new_inventory = self.repository.add_inventory_by_delta(basis_id, self._basis_delta, self._new_revision_id, self.parents)
        return self._new_revision_id

    def _gen_revision_id(self):
        """Return new revision-id."""
        return generate_ids.gen_revision_id(self._committer, self._timestamp)

    def _require_root_change(self, tree):
        """Enforce an appropriate root object change.

        This is called once when record_iter_changes is called, if and only if
        the root was not in the delta calculated by record_iter_changes.

        :param tree: The tree which is being committed.
        """
        if self.repository.supports_rich_root():
            return
        if len(self.parents) == 0:
            raise errors.RootMissing()
        entry = entry_factory['directory'](tree.path2id(''), '', None)
        entry.revision = self._new_revision_id
        self._basis_delta.append(('', '', entry.file_id, entry))

    def _get_delta(self, ie, basis_inv, path):
        """Get a delta against the basis inventory for ie."""
        if not basis_inv.has_id(ie.file_id):
            result = (None, path, ie.file_id, ie)
            self._basis_delta.append(result)
            return result
        elif ie != basis_inv.get_entry(ie.file_id):
            result = (basis_inv.id2path(ie.file_id), path, ie.file_id, ie)
            self._basis_delta.append(result)
            return result
        else:
            return None

    def _heads(self, file_id, revision_ids):
        """Calculate the graph heads for revision_ids in the graph of file_id.

        This can use either a per-file graph or a global revision graph as we
        have an identity relationship between the two graphs.
        """
        return self.__heads(revision_ids)

    def get_basis_delta(self):
        """Return the complete inventory delta versus the basis inventory.

        :return: An inventory delta, suitable for use with apply_delta, or
            Repository.add_inventory_by_delta, etc.
        """
        return self._basis_delta

    def record_iter_changes(self, tree, basis_revision_id, iter_changes, _entry_factory=entry_factory):
        """Record a new tree via iter_changes.

        :param tree: The tree to obtain text contents from for changed objects.
        :param basis_revision_id: The revision id of the tree the iter_changes
            has been generated against. Currently assumed to be the same
            as self.parents[0] - if it is not, errors may occur.
        :param iter_changes: An iter_changes iterator with the changes to apply
            to basis_revision_id. The iterator must not include any items with
            a current kind of None - missing items must be either filtered out
            or errored-on before record_iter_changes sees the item.
        :param _entry_factory: Private method to bind entry_factory locally for
            performance.
        :return: A generator of (relpath, fs_hash) tuples for use with
            tree._observed_sha1.
        """
        merged_ids = {}
        parent_entries = {}
        ghost_basis = False
        try:
            revtrees = list(self.repository.revision_trees(self.parents))
        except errors.NoSuchRevision:
            revtrees = []
            for revision_id in self.parents:
                try:
                    revtrees.append(self.repository.revision_tree(revision_id))
                except errors.NoSuchRevision:
                    if not revtrees:
                        basis_revision_id = _mod_revision.NULL_REVISION
                        ghost_basis = True
                    revtrees.append(self.repository.revision_tree(_mod_revision.NULL_REVISION))
        if revtrees:
            basis_tree = revtrees[0]
        else:
            basis_tree = self.repository.revision_tree(_mod_revision.NULL_REVISION)
        basis_inv = basis_tree.root_inventory
        if len(self.parents) > 0:
            if basis_revision_id != self.parents[0] and (not ghost_basis):
                raise Exception('arbitrary basis parents not yet supported with merges')
            for revtree in revtrees[1:]:
                for change in revtree.root_inventory._make_delta(basis_inv):
                    if change[1] is None:
                        continue
                    if change[2] not in merged_ids:
                        if change[0] is not None:
                            basis_entry = basis_inv.get_entry(change[2])
                            merged_ids[change[2]] = [basis_entry.revision, change[3].revision]
                            parent_entries[change[2]] = {basis_entry.revision: basis_entry, change[3].revision: change[3]}
                        else:
                            merged_ids[change[2]] = [change[3].revision]
                            parent_entries[change[2]] = {change[3].revision: change[3]}
                    else:
                        merged_ids[change[2]].append(change[3].revision)
                        parent_entries[change[2]][change[3].revision] = change[3]
        else:
            merged_ids = {}
        changes = {}
        for change in iter_changes:
            if change.path[0] is not None:
                head_candidate = [basis_inv.get_entry(change.file_id).revision]
            else:
                head_candidate = []
            changes[change.file_id] = (change, merged_ids.get(change.file_id, head_candidate))
        unchanged_merged = set(merged_ids) - set(changes)
        for file_id in unchanged_merged:
            try:
                basis_entry = basis_inv.get_entry(file_id)
            except errors.NoSuchId:
                pass
            else:
                change = InventoryTreeChange(file_id, (basis_inv.id2path(file_id), tree.id2path(file_id)), False, (True, True), (basis_entry.parent_id, basis_entry.parent_id), (basis_entry.name, basis_entry.name), (basis_entry.kind, basis_entry.kind), (basis_entry.executable, basis_entry.executable))
                changes[file_id] = (change, merged_ids[file_id])
        seen_root = False
        inv_delta = self._basis_delta
        modified_rev = self._new_revision_id
        for change, head_candidates in changes.values():
            if change.versioned[1]:
                kind = change.kind[1]
                file_id = change.file_id
                entry = _entry_factory[kind](file_id, change.name[1], change.parent_id[1])
                head_set = self._heads(change.file_id, set(head_candidates))
                heads = []
                for head_candidate in head_candidates:
                    if head_candidate in head_set:
                        heads.append(head_candidate)
                        head_set.remove(head_candidate)
                carried_over = False
                if len(heads) == 1:
                    parent_entry_revs = parent_entries.get(file_id, None)
                    if parent_entry_revs:
                        parent_entry = parent_entry_revs.get(heads[0], None)
                    else:
                        parent_entry = None
                    if parent_entry is None:
                        carry_over_possible = False
                    elif parent_entry.kind != entry.kind or parent_entry.parent_id != entry.parent_id or parent_entry.name != entry.name:
                        carry_over_possible = False
                    else:
                        carry_over_possible = True
                else:
                    carry_over_possible = False
                if kind == 'file':
                    if change.executable[1]:
                        entry.executable = True
                    else:
                        entry.executable = False
                    if carry_over_possible and parent_entry.executable == entry.executable:
                        nostore_sha = parent_entry.text_sha1
                    else:
                        nostore_sha = None
                    file_obj, stat_value = tree.get_file_with_stat(change.path[1])
                    try:
                        entry.text_sha1, entry.text_size = self._add_file_to_weave(file_id, file_obj, heads, nostore_sha, size=stat_value.st_size if stat_value else None)
                        yield (change.path[1], (entry.text_sha1, stat_value))
                    except versionedfile.ExistingContent:
                        carried_over = True
                        entry.text_size = parent_entry.text_size
                        entry.text_sha1 = parent_entry.text_sha1
                    finally:
                        file_obj.close()
                elif kind == 'symlink':
                    entry.symlink_target = tree.get_symlink_target(change.path[1])
                    if carry_over_possible and parent_entry.symlink_target == entry.symlink_target:
                        carried_over = True
                    else:
                        self._add_file_to_weave(change.file_id, BytesIO(), heads, None, size=0)
                elif kind == 'directory':
                    if carry_over_possible:
                        carried_over = True
                    elif change.path[1] != '' or self.repository.supports_rich_root():
                        self._add_file_to_weave(change.file_id, BytesIO(), heads, None, size=0)
                elif kind == 'tree-reference':
                    if not self.repository._format.supports_tree_reference:
                        raise errors.UnsupportedOperation(tree.add_reference, self.repository)
                    reference_revision = tree.get_reference_revision(change.path[1])
                    entry.reference_revision = reference_revision
                    if carry_over_possible and parent_entry.reference_revision == reference_revision:
                        carried_over = True
                    else:
                        self._add_file_to_weave(change.file_id, BytesIO(), heads, None, size=0)
                else:
                    raise AssertionError('unknown kind %r' % kind)
                if not carried_over:
                    entry.revision = modified_rev
                else:
                    entry.revision = parent_entry.revision
            else:
                entry = None
            new_path = change.path[1]
            inv_delta.append((change.path[0], new_path, change.file_id, entry))
            if new_path == '':
                seen_root = True
        if len(inv_delta) > 0 and basis_revision_id != _mod_revision.NULL_REVISION or (len(inv_delta) > 1 and basis_revision_id == _mod_revision.NULL_REVISION):
            self._any_changes = True
        if not seen_root:
            self._require_root_change(tree)
        self.basis_delta_revision = basis_revision_id

    def _add_file_to_weave(self, file_id, fileobj, parents, nostore_sha, size):
        parent_keys = tuple([(file_id, parent) for parent in parents])
        return self.repository.texts.add_content(versionedfile.FileContentFactory((file_id, self._new_revision_id), parent_keys, fileobj, size=size), nostore_sha=nostore_sha, random_id=self.random_revid)[0:2]