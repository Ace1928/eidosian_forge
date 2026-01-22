from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _plug_hole(self, hole, ancestors0, queue, potential_labels0, plug_acc0, record):
    """
        Try all possible ways of plugging a single hole.
        See _plug_nodes for the meanings of the parameters.
        """
    assert hole not in ancestors0
    ancestors = [hole] + ancestors0
    for l in potential_labels0:
        if self._violates_constraints(l, ancestors):
            continue
        plug_acc = plug_acc0.copy()
        plug_acc[hole] = l
        potential_labels = potential_labels0.copy()
        potential_labels.remove(l)
        if len(potential_labels) == 0:
            self._sanity_check_plugging(plug_acc, self.top_hole, [])
            record.append(plug_acc)
        else:
            self._plug_nodes(queue + [(l, ancestors)], potential_labels, plug_acc, record)