from nltk.cluster.util import Dendrogram, VectorSpaceClusterer, cosine_distance
def update_clusters(self, num_clusters):
    clusters = self._dendrogram.groups(num_clusters)
    self._centroids = []
    for cluster in clusters:
        assert len(cluster) > 0
        if self._should_normalise:
            centroid = self._normalise(cluster[0])
        else:
            centroid = numpy.array(cluster[0])
        for vector in cluster[1:]:
            if self._should_normalise:
                centroid += self._normalise(vector)
            else:
                centroid += vector
        centroid /= len(cluster)
        self._centroids.append(centroid)
    self._num_clusters = len(self._centroids)