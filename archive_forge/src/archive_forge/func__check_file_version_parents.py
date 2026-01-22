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
def _check_file_version_parents(self, texts, progress_bar):
    """See check_file_version_parents."""
    wrong_parents = {}
    self.file_ids = {file_id for file_id, _ in self.text_index}
    n_versions = len(self.text_index)
    progress_bar.update(gettext('loading text store'), 0, n_versions)
    parent_map = self.repository.texts.get_parent_map(self.text_index)
    text_keys = self.repository.texts.keys()
    unused_keys = frozenset(text_keys) - set(self.text_index)
    for num, key in enumerate(self.text_index):
        progress_bar.update(gettext('checking text graph'), num, n_versions)
        correct_parents = self.calculate_file_version_parents(key)
        try:
            knit_parents = parent_map[key]
        except errors.RevisionNotPresent:
            knit_parents = None
        if correct_parents != knit_parents:
            wrong_parents[key] = (knit_parents, correct_parents)
    return (wrong_parents, unused_keys)