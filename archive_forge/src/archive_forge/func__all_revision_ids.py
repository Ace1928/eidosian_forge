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
def _all_revision_ids(self):
    """Returns a list of all the revision ids in the repository.

        These are in as much topological order as the underlying store can
        present: for weaves ghosts may lead to a lack of correctness until
        the reweave updates the parents list.
        """
    with self.lock_read():
        return [key[-1] for key in self.revisions.keys()]