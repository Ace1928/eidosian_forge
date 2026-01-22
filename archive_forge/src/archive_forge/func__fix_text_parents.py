from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _fix_text_parents(self):
    """Fix bad versionedfile parent entries.

        It is possible for the parents entry in a versionedfile entry to be
        inconsistent with the values in the revision and inventory.

        This method finds entries with such inconsistencies, corrects their
        parent lists, and replaces the versionedfile with a corrected version.
        """
    transaction = self.repo.get_transaction()
    versions = [key[-1] for key in self.revisions.keys()]
    mutter('Prepopulating revision text cache with %d revisions', len(versions))
    vf_checker = self.repo._get_versioned_file_checker()
    bad_parents, unused_versions = vf_checker.check_file_version_parents(self.repo.texts, self.pb)
    text_index = vf_checker.text_index
    per_id_bad_parents = {}
    for key in unused_versions:
        per_id_bad_parents[key[0]] = {}
    for key, details in bad_parents.items():
        file_id = key[0]
        rev_id = key[1]
        knit_parents = tuple([parent[-1] for parent in details[0]])
        correct_parents = tuple([parent[-1] for parent in details[1]])
        file_details = per_id_bad_parents.setdefault(file_id, {})
        file_details[rev_id] = (knit_parents, correct_parents)
    file_id_versions = {}
    for text_key in text_index:
        versions_list = file_id_versions.setdefault(text_key[0], [])
        versions_list.append(text_key[1])
    for num, file_id in enumerate(per_id_bad_parents):
        self.pb.update(gettext('Fixing text parents'), num, len(per_id_bad_parents))
        versions_with_bad_parents = per_id_bad_parents[file_id]
        id_unused_versions = {key[-1] for key in unused_versions if key[0] == file_id}
        if file_id in file_id_versions:
            file_versions = file_id_versions[file_id]
        else:
            file_versions = []
        self._fix_text_parent(file_id, versions_with_bad_parents, id_unused_versions, file_versions)