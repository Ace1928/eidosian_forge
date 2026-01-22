from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def _copy_nodes(self, nodes, index_map, writer, write_index, output_lines=None):
    """Copy knit nodes between packs with no graph references.

        :param output_lines: Output full texts of copied items.
        """
    with ui.ui_factory.nested_progress_bar() as pb:
        return self._do_copy_nodes(nodes, index_map, writer, write_index, pb, output_lines=output_lines)