from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _fix_text_parent(self, file_id, versions_with_bad_parents, unused_versions, all_versions):
    """Fix bad versionedfile entries in a single versioned file."""
    mutter('fixing text parent: %r (%d versions)', file_id, len(versions_with_bad_parents))
    mutter('(%d are unused)', len(unused_versions))
    new_file_id = b'temp:%s' % file_id
    new_parents = {}
    needed_keys = set()
    for version in all_versions:
        if version in unused_versions:
            continue
        elif version in versions_with_bad_parents:
            parents = versions_with_bad_parents[version][1]
        else:
            pmap = self.repo.texts.get_parent_map([(file_id, version)])
            parents = [key[-1] for key in pmap[file_id, version]]
        new_parents[new_file_id, version] = [(new_file_id, parent) for parent in parents]
        needed_keys.add((file_id, version))

    def fix_parents(stream):
        for record in stream:
            chunks = record.get_bytes_as('chunked')
            new_key = (new_file_id, record.key[-1])
            parents = new_parents[new_key]
            yield ChunkedContentFactory(new_key, parents, record.sha1, chunks)
    stream = self.repo.texts.get_record_stream(needed_keys, 'topological', True)
    self.repo._remove_file_id(new_file_id)
    self.repo.texts.insert_record_stream(fix_parents(stream))
    self.repo._remove_file_id(file_id)
    if len(new_parents):
        self.repo._move_file_id(new_file_id, file_id)