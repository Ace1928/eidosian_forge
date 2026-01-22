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
class InterSameDataRepository(InterVersionedFileRepository):
    """Code for converting between repositories that represent the same data.

    Data format and model must match for this to work.
    """

    @classmethod
    def _get_repo_format_to_test(self):
        """Repository format for testing with.

        InterSameData can pull from subtree to subtree and from non-subtree to
        non-subtree, so we test this with the richest repository format.
        """
        from breezy.bzr import knitrepo
        return knitrepo.RepositoryFormatKnit3()

    @staticmethod
    def is_compatible(source, target):
        return InterRepository._same_model(source, target) and source._format.supports_full_versioned_files and target._format.supports_full_versioned_files