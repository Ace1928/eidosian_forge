from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _plug_nodes(self, queue, potential_labels, plug_acc, record):
    """
        Plug the nodes in `queue' with the labels in `potential_labels'.

        Each element of `queue' is a tuple of the node to plug and the list of
        ancestor holes from the root of the graph to that node.

        `potential_labels' is a set of the labels which are still available for
        plugging.

        `plug_acc' is the incomplete mapping of holes to labels made on the
        current branch of the search tree so far.

        `record' is a list of all the complete pluggings that we have found in
        total so far.  It is the only parameter that is destructively updated.
        """
    if queue != []:
        node, ancestors = queue[0]
        if node in self.holes:
            self._plug_hole(node, ancestors, queue[1:], potential_labels, plug_acc, record)
        else:
            assert node in self.labels
            args = self.fragments[node][1]
            head = [(a, ancestors) for a in args if self.is_node(a)]
            self._plug_nodes(head + queue[1:], potential_labels, plug_acc, record)
    else:
        raise Exception('queue empty')