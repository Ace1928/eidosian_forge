import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
def upgma(self, distance_matrix):
    """Construct and return an UPGMA tree.

        Constructs and returns an Unweighted Pair Group Method
        with Arithmetic mean (UPGMA) tree.

        :Parameters:
            distance_matrix : DistanceMatrix
                The distance matrix for tree construction.

        """
    if not isinstance(distance_matrix, DistanceMatrix):
        raise TypeError('Must provide a DistanceMatrix object.')
    dm = copy.deepcopy(distance_matrix)
    clades = [BaseTree.Clade(None, name) for name in dm.names]
    min_i = 0
    min_j = 0
    inner_count = 0
    while len(dm) > 1:
        min_dist = dm[1, 0]
        for i in range(1, len(dm)):
            for j in range(0, i):
                if min_dist >= dm[i, j]:
                    min_dist = dm[i, j]
                    min_i = i
                    min_j = j
        clade1 = clades[min_i]
        clade2 = clades[min_j]
        inner_count += 1
        inner_clade = BaseTree.Clade(None, 'Inner' + str(inner_count))
        inner_clade.clades.append(clade1)
        inner_clade.clades.append(clade2)
        if clade1.is_terminal():
            clade1.branch_length = min_dist / 2
        else:
            clade1.branch_length = min_dist / 2 - self._height_of(clade1)
        if clade2.is_terminal():
            clade2.branch_length = min_dist / 2
        else:
            clade2.branch_length = min_dist / 2 - self._height_of(clade2)
        clades[min_j] = inner_clade
        del clades[min_i]
        for k in range(0, len(dm)):
            if k != min_i and k != min_j:
                dm[min_j, k] = (dm[min_i, k] + dm[min_j, k]) / 2
        dm.names[min_j] = 'Inner' + str(inner_count)
        del dm[min_i]
    inner_clade.branch_length = 0
    return BaseTree.Tree(inner_clade)