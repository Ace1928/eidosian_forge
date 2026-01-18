import numbers
from . import _cluster  # type: ignore
def treecluster(self, transpose=False, method='m', dist='e'):
    """Apply hierarchical clustering and return a Tree object.

        The pairwise single, complete, centroid, and average linkage
        hierarchical clustering methods are available.

        Keyword arguments:
         - transpose: if False: rows are clustered;
                      if True: columns are clustered.
         - dist: specifies the distance function to be used:
           - dist == 'e': Euclidean distance
           - dist == 'b': City Block distance
           - dist == 'c': Pearson correlation
           - dist == 'a': absolute value of the correlation
           - dist == 'u': uncentered correlation
           - dist == 'x': absolute uncentered correlation
           - dist == 's': Spearman's rank correlation
           - dist == 'k': Kendall's tau
         - method: specifies which linkage method is used:
           - method == 's': Single pairwise linkage
           - method == 'm': Complete (maximum) pairwise linkage (default)
           - method == 'c': Centroid linkage
           - method == 'a': Average pairwise linkage

        See the description of the Tree class for more information about
        the Tree object returned by this method.
        """
    if transpose:
        weight = self.gweight
    else:
        weight = self.eweight
    return treecluster(self.data, self.mask, weight, transpose, method, dist)