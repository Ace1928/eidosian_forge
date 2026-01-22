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
def _iter_inventory_xmls(self, revision_ids, ordering):
    if ordering is None:
        order_as_requested = True
        ordering = 'unordered'
    else:
        order_as_requested = False
    keys = [(revision_id,) for revision_id in revision_ids]
    if not keys:
        return
    if order_as_requested:
        key_iter = iter(keys)
        next_key = next(key_iter)
    stream = self.inventories.get_record_stream(keys, ordering, True)
    text_lines = {}
    for record in stream:
        if record.storage_kind != 'absent':
            lines = record.get_bytes_as('lines')
            if order_as_requested:
                text_lines[record.key] = lines
            else:
                yield (lines, record.key[-1])
        else:
            yield (None, record.key[-1])
        if order_as_requested:
            while next_key in text_lines:
                lines = text_lines.pop(next_key)
                yield (lines, next_key[-1])
                try:
                    next_key = next(key_iter)
                except StopIteration:
                    next_key = None
                    break