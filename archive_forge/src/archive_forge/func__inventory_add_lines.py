import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
def _inventory_add_lines(self, revision_id, parents, lines, check_content=True):
    """Store lines in inv_vf and return the sha1 of the inventory."""
    present_parents = self.get_graph().get_parent_map(parents)
    final_parents = []
    for parent in parents:
        if parent in present_parents:
            final_parents.append((parent,))
    return self.inventories.add_lines((revision_id,), final_parents, lines, check_content=check_content)[0]