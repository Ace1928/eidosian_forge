from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
class VersionedFileCheck(Check):
    """Check a versioned file repository"""

    def __init__(self, repository, check_repo=True):
        self.repository = repository
        self.checked_rev_cnt = 0
        self.ghosts = set()
        self.missing_parent_links = {}
        self.missing_inventory_sha_cnt = 0
        self.missing_revision_cnt = 0
        self.checked_weaves = set()
        self.unreferenced_versions = set()
        self.inconsistent_parents = []
        self.rich_roots = repository.supports_rich_root()
        self.text_key_references = {}
        self.check_repo = check_repo
        self.other_results = []
        self._report_items = []
        self.pending_keys = {}
        self.ancestors = {}

    def check(self, callback_refs=None, check_repo=True):
        if callback_refs is None:
            callback_refs = {}
        with self.repository.lock_read(), ui.ui_factory.nested_progress_bar() as self.progress:
            self.progress.update(gettext('check'), 0, 4)
            if self.check_repo:
                self.progress.update(gettext('checking revisions'), 0)
                self.check_revisions()
                self.progress.update(gettext('checking commit contents'), 1)
                self.repository._check_inventories(self)
                self.progress.update(gettext('checking file graphs'), 2)
                self.check_weaves()
            self.progress.update(gettext('checking branches and trees'), 3)
            if callback_refs:
                repo = self.repository
                refs = {}
                wanting_items = set()
                distances = set()
                existences = set()
                for ref, wantlist in callback_refs.items():
                    wanting_items.update(wantlist)
                    kind, value = ref
                    if kind == 'trees':
                        refs[ref] = repo.revision_tree(value)
                    elif kind == 'lefthand-distance':
                        distances.add(value)
                    elif kind == 'revision-existence':
                        existences.add(value)
                    else:
                        raise AssertionError('unknown ref kind for ref %s' % ref)
                node_distances = repo.get_graph().find_lefthand_distances(distances)
                for key, distance in node_distances.items():
                    refs['lefthand-distance', key] = distance
                    if key in existences and distance > 0:
                        refs['revision-existence', key] = True
                        existences.remove(key)
                parent_map = repo.get_graph().get_parent_map(existences)
                for key in parent_map:
                    refs['revision-existence', key] = True
                    existences.remove(key)
                for key in existences:
                    refs['revision-existence', key] = False
                for item in wanting_items:
                    if isinstance(item, WorkingTree):
                        item._check(refs)
                    if isinstance(item, Branch):
                        self.other_results.append(item.check(refs))

    def _check_revisions(self, revisions_iterator):
        """Check revision objects by decorating a generator.

        :param revisions_iterator: An iterator of(revid, Revision-or-None).
        :return: A generator of the contents of revisions_iterator.
        """
        self.planned_revisions = set()
        for revid, revision in revisions_iterator:
            yield (revid, revision)
            self._check_one_rev(revid, revision)
        self.planned_revisions = list(self.planned_revisions)

    def check_revisions(self):
        """Scan revisions, checking data directly available as we go."""
        revision_iterator = self.repository.iter_revisions(self.repository.all_revision_ids())
        revision_iterator = self._check_revisions(revision_iterator)
        if not self.repository._format.revision_graph_can_have_wrong_parents:
            self.revs_with_bad_parents_in_index = None
            for thing in revision_iterator:
                pass
        else:
            bad_revisions = self.repository._find_inconsistent_revision_parents(revision_iterator)
            self.revs_with_bad_parents_in_index = list(bad_revisions)

    def report_results(self, verbose):
        if self.check_repo:
            self._report_repo_results(verbose)
        for result in self.other_results:
            result.report_results(verbose)

    def _report_repo_results(self, verbose):
        note(gettext('checked repository {0} format {1}').format(self.repository.user_url, self.repository._format))
        note(gettext('%6d revisions'), self.checked_rev_cnt)
        note(gettext('%6d file-ids'), len(self.checked_weaves))
        if verbose:
            note(gettext('%6d unreferenced text versions'), len(self.unreferenced_versions))
        if verbose and len(self.unreferenced_versions):
            for file_id, revision_id in self.unreferenced_versions:
                note(gettext('unreferenced version: {{{0}}} in {1}').format(revision_id.decode('utf-8'), file_id.decode('utf-8')))
        if self.missing_inventory_sha_cnt:
            note(gettext('%6d revisions are missing inventory_sha1'), self.missing_inventory_sha_cnt)
        if self.missing_revision_cnt:
            note(gettext('%6d revisions are mentioned but not present'), self.missing_revision_cnt)
        if len(self.ghosts):
            note(gettext('%6d ghost revisions'), len(self.ghosts))
            if verbose:
                for ghost in self.ghosts:
                    note('      %s', ghost.decode('utf-8'))
        if len(self.missing_parent_links):
            note(gettext('%6d revisions missing parents in ancestry'), len(self.missing_parent_links))
            if verbose:
                for link, linkers in self.missing_parent_links.items():
                    note(gettext('      %s should be in the ancestry for:'), link.decode('utf-8'))
                    for linker in linkers:
                        note('       * %s', linker.decode('utf-8'))
        if len(self.inconsistent_parents):
            note(gettext('%6d inconsistent parents'), len(self.inconsistent_parents))
            if verbose:
                for info in self.inconsistent_parents:
                    revision_id, file_id, found_parents, correct_parents = info
                    note(gettext('      * {0} version {1} has parents ({2}) but should have ({3})').format(file_id.decode('utf-8'), revision_id.decode('utf-8'), ', '.join((p.decode('utf-8') for p in found_parents)), ', '.join((p.decode('utf-8') for p in correct_parents))))
        if self.revs_with_bad_parents_in_index:
            note(gettext('%6d revisions have incorrect parents in the revision index'), len(self.revs_with_bad_parents_in_index))
            if verbose:
                for item in self.revs_with_bad_parents_in_index:
                    revision_id, index_parents, actual_parents = item
                    note(gettext('       {0} has wrong parents in index: ({1}) should be ({2})').format(revision_id.decode('utf-8'), ', '.join((p.decode('utf-8') for p in index_parents)), ', '.join((p.decode('utf-8') for p in actual_parents))))
        for item in self._report_items:
            note(item)

    def _check_one_rev(self, rev_id, rev):
        """Cross-check one revision.

        :param rev_id: A revision id to check.
        :param rev: A revision or None to indicate a missing revision.
        """
        if rev.revision_id != rev_id:
            self._report_items.append(gettext('Mismatched internal revid {{{0}}} and index revid {{{1}}}').format(rev.revision_id.decode('utf-8'), rev_id.decode('utf-8')))
            rev_id = rev.revision_id
        self.planned_revisions.add(rev_id)
        self.ghosts.discard(rev_id)
        for parent in rev.parent_ids:
            if parent not in self.planned_revisions:
                self.ghosts.add(parent)
        self.ancestors[rev_id] = tuple(rev.parent_ids) or (NULL_REVISION,)
        self.add_pending_item(rev_id, ('inventories', rev_id), 'inventory', rev.inventory_sha1)
        self.checked_rev_cnt += 1

    def add_pending_item(self, referer, key, kind, sha1):
        """Add a reference to a sha1 to be cross checked against a key.

        :param referer: The referer that expects key to have sha1.
        :param key: A storage key e.g. ('texts', 'foo@bar-20040504-1234')
        :param kind: revision/inventory/text/map/signature
        :param sha1: A hex sha1 or None if no sha1 is known.
        """
        existing = self.pending_keys.get(key)
        if existing:
            if sha1 != existing[1]:
                self._report_items.append(gettext('Multiple expected sha1s for {0}. {{{1}}} expects {{{2}}}, {{{3}}} expects {{{4}}}').format(key, referer, sha1, existing[1], existing[0]))
        else:
            self.pending_keys[key] = (kind, sha1, referer)

    def check_weaves(self):
        """Check all the weaves we can get our hands on.
        """
        weave_ids = []
        with ui.ui_factory.nested_progress_bar() as storebar:
            self._check_weaves(storebar)

    def _check_weaves(self, storebar):
        storebar.update('text-index', 0, 2)
        if self.repository._format.fast_deltas:
            weave_checker = self.repository._get_versioned_file_checker(ancestors=self.ancestors)
        else:
            weave_checker = self.repository._get_versioned_file_checker(text_key_references=self.text_key_references, ancestors=self.ancestors)
        storebar.update('file-graph', 1)
        wrongs, unused_versions = weave_checker.check_file_version_parents(self.repository.texts)
        self.checked_weaves = weave_checker.file_ids
        for text_key, (stored_parents, correct_parents) in wrongs.items():
            weave_id = text_key[0]
            revision_id = text_key[-1]
            weave_parents = tuple([parent[-1] for parent in stored_parents])
            correct_parents = tuple([parent[-1] for parent in correct_parents])
            self.inconsistent_parents.append((revision_id, weave_id, weave_parents, correct_parents))
        self.unreferenced_versions.update(unused_versions)

    def _add_entry_to_text_key_references(self, inv, entry):
        if not self.rich_roots and entry.name == '':
            return
        key = (entry.file_id, entry.revision)
        self.text_key_references.setdefault(key, False)
        if entry.revision == inv.revision_id:
            self.text_key_references[key] = True