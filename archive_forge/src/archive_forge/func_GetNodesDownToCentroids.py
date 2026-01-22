def GetNodesDownToCentroids(cluster, above=1):
    """returns an ordered list of all nodes below cluster


  """
    if hasattr(cluster, '_isCentroid'):
        cluster._aboveCentroid = 0
        above = -1
    else:
        cluster._aboveCentroid = above
    if len(cluster) == 1:
        return [cluster]
    else:
        res = []
        children = cluster.GetChildren()
        children.sort(key=lambda x: len(x), reverse=True)
        for child in children:
            res = res + GetNodesDownToCentroids(child, above)
        res = res + [cluster]
        return res