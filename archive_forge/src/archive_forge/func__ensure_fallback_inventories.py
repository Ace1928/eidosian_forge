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