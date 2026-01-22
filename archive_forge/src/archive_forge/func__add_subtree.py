import random
import sys
from . import Nodes
def _add_subtree(self, parent_id=None, tree=None):
    """Add leaf or tree (in newick format) to a parent_id (PRIVATE)."""
    if parent_id is None:
        raise TreeError('Need node_id to connect to.')
    for st in tree:
        nd = self.dataclass()
        nd = self._add_nodedata(nd, st)
        if isinstance(st[0], list):
            sn = Nodes.Node(nd)
            self.add(sn, parent_id)
            self._add_subtree(sn.id, st[0])
        else:
            nd.taxon = st[0]
            leaf = Nodes.Node(nd)
            self.add(leaf, parent_id)