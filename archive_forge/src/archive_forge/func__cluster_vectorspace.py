import copy
import random
import sys
from nltk.cluster.util import VectorSpaceClusterer
def _cluster_vectorspace(self, vectors, trace=False):
    if self._num_means < len(vectors):
        converged = False
        while not converged:
            clusters = [[] for m in range(self._num_means)]
            for vector in vectors:
                index = self.classify_vectorspace(vector)
                clusters[index].append(vector)
            if trace:
                print('iteration')
            new_means = list(map(self._centroid, clusters, self._means))
            difference = self._sum_distances(self._means, new_means)
            if difference < self._max_difference:
                converged = True
            self._means = new_means