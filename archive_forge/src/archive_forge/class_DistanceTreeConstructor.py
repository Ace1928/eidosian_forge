import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class DistanceTreeConstructor(TreeConstructor):
    """Distance based tree constructor.

    :Parameters:
        method : str
            Distance tree construction method, 'nj'(default) or 'upgma'.
        distance_calculator : DistanceCalculator
            The distance matrix calculator for multiple sequence alignment.
            It must be provided if ``build_tree`` will be called.

    Examples
    --------
    Loading a small PHYLIP alignment from which to compute distances, and then
    build a upgma Tree::

      >>> from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
      >>> from Bio.Phylo.TreeConstruction import DistanceCalculator
      >>> from Bio import AlignIO
      >>> aln = AlignIO.read(open('TreeConstruction/msa.phy'), 'phylip')
      >>> constructor = DistanceTreeConstructor()
      >>> calculator = DistanceCalculator('identity')
      >>> dm = calculator.get_distance(aln)
      >>> upgmatree = constructor.upgma(dm)
      >>> print(upgmatree)
      Tree(rooted=True)
          Clade(branch_length=0, name='Inner4')
              Clade(branch_length=0.18749999999999994, name='Inner1')
                  Clade(branch_length=0.07692307692307693, name='Epsilon')
                  Clade(branch_length=0.07692307692307693, name='Delta')
              Clade(branch_length=0.11057692307692304, name='Inner3')
                  Clade(branch_length=0.038461538461538464, name='Inner2')
                      Clade(branch_length=0.11538461538461536, name='Gamma')
                      Clade(branch_length=0.11538461538461536, name='Beta')
                  Clade(branch_length=0.15384615384615383, name='Alpha')

    Build a NJ Tree::

      >>> njtree = constructor.nj(dm)
      >>> print(njtree)
      Tree(rooted=False)
          Clade(branch_length=0, name='Inner3')
              Clade(branch_length=0.18269230769230765, name='Alpha')
              Clade(branch_length=0.04807692307692307, name='Beta')
              Clade(branch_length=0.04807692307692307, name='Inner2')
                  Clade(branch_length=0.27884615384615385, name='Inner1')
                      Clade(branch_length=0.051282051282051266, name='Epsilon')
                      Clade(branch_length=0.10256410256410259, name='Delta')
                  Clade(branch_length=0.14423076923076922, name='Gamma')

    Same example, using the new Alignment class::

      >>> from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
      >>> from Bio.Phylo.TreeConstruction import DistanceCalculator
      >>> from Bio import Align
      >>> aln = Align.read(open('TreeConstruction/msa.phy'), 'phylip')
      >>> constructor = DistanceTreeConstructor()
      >>> calculator = DistanceCalculator('identity')
      >>> dm = calculator.get_distance(aln)
      >>> upgmatree = constructor.upgma(dm)
      >>> print(upgmatree)
      Tree(rooted=True)
          Clade(branch_length=0, name='Inner4')
              Clade(branch_length=0.18749999999999994, name='Inner1')
                  Clade(branch_length=0.07692307692307693, name='Epsilon')
                  Clade(branch_length=0.07692307692307693, name='Delta')
              Clade(branch_length=0.11057692307692304, name='Inner3')
                  Clade(branch_length=0.038461538461538464, name='Inner2')
                      Clade(branch_length=0.11538461538461536, name='Gamma')
                      Clade(branch_length=0.11538461538461536, name='Beta')
                  Clade(branch_length=0.15384615384615383, name='Alpha')

    Build a NJ Tree::

      >>> njtree = constructor.nj(dm)
      >>> print(njtree)
      Tree(rooted=False)
          Clade(branch_length=0, name='Inner3')
              Clade(branch_length=0.18269230769230765, name='Alpha')
              Clade(branch_length=0.04807692307692307, name='Beta')
              Clade(branch_length=0.04807692307692307, name='Inner2')
                  Clade(branch_length=0.27884615384615385, name='Inner1')
                      Clade(branch_length=0.051282051282051266, name='Epsilon')
                      Clade(branch_length=0.10256410256410259, name='Delta')
                  Clade(branch_length=0.14423076923076922, name='Gamma')

    """
    methods = ['nj', 'upgma']

    def __init__(self, distance_calculator=None, method='nj'):
        """Initialize the class."""
        if distance_calculator is None or isinstance(distance_calculator, DistanceCalculator):
            self.distance_calculator = distance_calculator
        else:
            raise TypeError('Must provide a DistanceCalculator object.')
        if method in self.methods:
            self.method = method
        else:
            raise TypeError('Bad method: ' + method + '. Available methods: ' + ', '.join(self.methods))

    def build_tree(self, msa):
        """Construct and return a Tree, Neighbor Joining or UPGMA."""
        if self.distance_calculator:
            dm = self.distance_calculator.get_distance(msa)
            tree = None
            if self.method == 'upgma':
                tree = self.upgma(dm)
            else:
                tree = self.nj(dm)
            return tree
        else:
            raise TypeError('Must provide a DistanceCalculator object.')

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

    def nj(self, distance_matrix):
        """Construct and return a Neighbor Joining tree.

        :Parameters:
            distance_matrix : DistanceMatrix
                The distance matrix for tree construction.

        """
        if not isinstance(distance_matrix, DistanceMatrix):
            raise TypeError('Must provide a DistanceMatrix object.')
        dm = copy.deepcopy(distance_matrix)
        clades = [BaseTree.Clade(None, name) for name in dm.names]
        node_dist = [0] * len(dm)
        min_i = 0
        min_j = 0
        inner_count = 0
        if len(dm) == 1:
            root = clades[0]
            return BaseTree.Tree(root, rooted=False)
        elif len(dm) == 2:
            min_i = 1
            min_j = 0
            clade1 = clades[min_i]
            clade2 = clades[min_j]
            clade1.branch_length = dm[min_i, min_j] / 2.0
            clade2.branch_length = dm[min_i, min_j] - clade1.branch_length
            inner_clade = BaseTree.Clade(None, 'Inner')
            inner_clade.clades.append(clade1)
            inner_clade.clades.append(clade2)
            clades[0] = inner_clade
            root = clades[0]
            return BaseTree.Tree(root, rooted=False)
        while len(dm) > 2:
            for i in range(0, len(dm)):
                node_dist[i] = 0
                for j in range(0, len(dm)):
                    node_dist[i] += dm[i, j]
                node_dist[i] = node_dist[i] / (len(dm) - 2)
            min_dist = dm[1, 0] - node_dist[1] - node_dist[0]
            min_i = 0
            min_j = 1
            for i in range(1, len(dm)):
                for j in range(0, i):
                    temp = dm[i, j] - node_dist[i] - node_dist[j]
                    if min_dist > temp:
                        min_dist = temp
                        min_i = i
                        min_j = j
            clade1 = clades[min_i]
            clade2 = clades[min_j]
            inner_count += 1
            inner_clade = BaseTree.Clade(None, 'Inner' + str(inner_count))
            inner_clade.clades.append(clade1)
            inner_clade.clades.append(clade2)
            clade1.branch_length = (dm[min_i, min_j] + node_dist[min_i] - node_dist[min_j]) / 2.0
            clade2.branch_length = dm[min_i, min_j] - clade1.branch_length
            clades[min_j] = inner_clade
            del clades[min_i]
            for k in range(0, len(dm)):
                if k != min_i and k != min_j:
                    dm[min_j, k] = (dm[min_i, k] + dm[min_j, k] - dm[min_i, min_j]) / 2.0
            dm.names[min_j] = 'Inner' + str(inner_count)
            del dm[min_i]
        root = None
        if clades[0] == inner_clade:
            clades[0].branch_length = 0
            clades[1].branch_length = dm[1, 0]
            clades[0].clades.append(clades[1])
            root = clades[0]
        else:
            clades[0].branch_length = dm[1, 0]
            clades[1].branch_length = 0
            clades[1].clades.append(clades[0])
            root = clades[1]
        return BaseTree.Tree(root, rooted=False)

    def _height_of(self, clade):
        """Calculate clade height -- the longest path to any terminal (PRIVATE)."""
        height = 0
        if clade.is_terminal():
            height = clade.branch_length
        else:
            height = height + max((self._height_of(c) for c in clade.clades))
        return height