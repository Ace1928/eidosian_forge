import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def write_revisions(self):
    """Write bundle records for all revisions and signatures"""
    inv_vf = self.repository.inventories
    topological_order = [key[-1] for key in multiparent.topo_iter_keys(inv_vf, self.revision_keys)]
    revision_order = topological_order
    if self.target is not None and self.target in self.revision_ids:
        revision_order = list(topological_order)
        revision_order.remove(self.target)
        revision_order.append(self.target)
    if self.repository._serializer.support_altered_by_hack:
        self._add_mp_records_keys('inventory', inv_vf, [(revid,) for revid in topological_order])
    else:
        self._add_inventory_mpdiffs_from_serializer(topological_order)
    self._add_revision_texts(revision_order)