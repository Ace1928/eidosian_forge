import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _is_tree(entry):
    mode = entry.mode
    if mode is None:
        return False
    return stat.S_ISDIR(mode)