import os
from ... import config as _mod_config
from ... import osutils, ui
from ...bzr.generate_ids import gen_revision_id
from ...bzr.inventorytree import InventoryTreeChange
from ...errors import (BzrError, NoCommonAncestor, UnknownFormatError,
from ...graph import FrozenHeadsCache
from ...merge import Merger
from ...revision import NULL_REVISION
from ...trace import mutter
from ...transport import NoSuchFile
from ...tsort import topo_sort
from .maptree import MapTree, map_file_ids
def unmarshall_rebase_plan(text):
    """Unmarshall a rebase plan.

    :param text: Text to parse
    :return: Tuple with last revision info, replace map.
    """
    lines = text.split(b'\n')
    if lines[0] != b'# Bazaar rebase plan %d' % REBASE_PLAN_VERSION:
        raise UnknownFormatError(lines[0])
    pts = lines[1].split(b' ', 1)
    last_revision_info = (int(pts[0]), pts[1])
    replace_map = {}
    for l in lines[2:]:
        if l == b'':
            continue
        pts = l.split(b' ')
        replace_map[pts[0]] = (pts[1], tuple(pts[2:]))
    return (last_revision_info, replace_map)