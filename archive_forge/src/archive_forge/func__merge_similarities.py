from nltk.cluster.util import Dendrogram, VectorSpaceClusterer, cosine_distance
def _merge_similarities(self, dist, cluster_len, i, j):
    i_weight = cluster_len[i]
    j_weight = cluster_len[j]
    weight_sum = i_weight + j_weight
    dist[:i, i] = dist[:i, i] * i_weight + dist[:i, j] * j_weight
    dist[:i, i] /= weight_sum
    dist[i, i + 1:j] = dist[i, i + 1:j] * i_weight + dist[i + 1:j, j] * j_weight
    dist[i, j + 1:] = dist[i, j + 1:] * i_weight + dist[j, j + 1:] * j_weight
    dist[i, i + 1:] /= weight_sum