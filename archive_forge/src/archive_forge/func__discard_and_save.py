from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _discard_and_save(self, packs):
    """Discard some packs from the repository.

        This removes them from the memory index, saves the in-memory index
        which makes the newly reconciled pack visible and hides the packs to be
        discarded, and finally renames the packs being discarded into the
        obsolete packs directory.

        :param packs: The packs to discard.
        """
    for pack in packs:
        self.repo._pack_collection._remove_pack_from_memory(pack)
    self.repo._pack_collection._save_pack_names()
    self.repo._pack_collection._obsolete_packs(packs)