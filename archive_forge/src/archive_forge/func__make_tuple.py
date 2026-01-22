from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
def _make_tuple(T, root, _parent):
    """Recursively compute the nested tuple representation of the
        given rooted tree.

        ``_parent`` is the parent node of ``root`` in the supertree in
        which ``T`` is a subtree, or ``None`` if ``root`` is the root of
        the supertree. This argument is used to determine which
        neighbors of ``root`` are children and which is the parent.

        """
    children = set(T[root]) - {_parent}
    if len(children) == 0:
        return ()
    nested = (_make_tuple(T, v, root) for v in children)
    if canonical_form:
        nested = sorted(nested)
    return tuple(nested)