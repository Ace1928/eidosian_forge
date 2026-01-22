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
def _copy_nodes_graph(self, index_map, writer, write_index, readv_group_iter, total_items, output_lines=False):
    """Copy knit nodes between packs.

        :param output_lines: Return lines present in the copied data as
            an iterator of line,version_id.
        """
    with ui.ui_factory.nested_progress_bar() as pb:
        yield from self._do_copy_nodes_graph(index_map, writer, write_index, output_lines, pb, readv_group_iter, total_items)