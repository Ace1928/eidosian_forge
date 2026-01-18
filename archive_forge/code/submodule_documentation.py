from typing import Iterator, Tuple
from .object_store import iter_tree_contents
from .objects import S_ISGITLINK
Iterate over cached submodules.

    Args:
      store: Object store to iterate
      root_tree_id: SHA of root tree

    Returns:
      Iterator over over (path, sha) tuples
    