import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _skip_tree(entry, include_trees):
    if entry.mode is None or (not include_trees and stat.S_ISDIR(entry.mode)):
        return _NULL_ENTRY
    return entry