def _HeightFirstSplit(cluster, n):
    """  *Internal Use Only*

  """
    if len(cluster) < n:
        raise ValueError('Cannot split cluster of length %d into %d pieces' % (len(cluster), n))
    if len(cluster) == n:
        return cluster.GetPoints()
    clusters = [cluster]
    for _ in range(n - 1):
        nxtIdx = 0
        while nxtIdx < len(clusters) and len(clusters[nxtIdx]) == 1:
            nxtIdx += 1
        assert nxtIdx < len(clusters)
        children = clusters[nxtIdx].GetChildren()
        for child in children:
            clusters.append(child)
        del clusters[nxtIdx]
        clusters.sort(key=lambda x: x.GetMetric(), reverse=True)
    return clusters