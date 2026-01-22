import itertools
def OrderTriangle(featIndices, dists):
    """
      put the distances for a triangle into canonical order

      It's easy if the features are all different:

      >>> OrderTriangle([0, 2, 4], [1, 2, 3])
      ([0, 2, 4], [1, 2, 3])

      It's trickiest if they are all the same:
      
      >>> OrderTriangle([0, 0, 0], [1, 2, 3])
      ([0, 0, 0], [3, 2, 1])
      >>> OrderTriangle([0, 0, 0], [2, 1, 3])
      ([0, 0, 0], [3, 2, 1])
      >>> OrderTriangle([0, 0, 0], [1, 3, 2])
      ([0, 0, 0], [3, 2, 1])
      >>> OrderTriangle([0, 0, 0], [3, 1, 2])
      ([0, 0, 0], [3, 2, 1])
      >>> OrderTriangle([0, 0, 0], [3, 2, 1])
      ([0, 0, 0], [3, 2, 1])

      >>> OrderTriangle([0, 0, 1], [3, 2, 1])
      ([0, 0, 1], [3, 2, 1])
      >>> OrderTriangle([0, 0, 1], [1, 3, 2])
      ([0, 0, 1], [1, 3, 2])
      >>> OrderTriangle([0, 0, 1], [1, 2, 3])
      ([0, 0, 1], [1, 3, 2])

    """
    if len(featIndices) != 3:
        raise ValueError('bad indices')
    if len(dists) != 3:
        raise ValueError('bad dists')
    fs = set(featIndices)
    if len(fs) == 3:
        return (featIndices, dists)
    dSums = [0] * 3
    dSums[0] = dists[0] + dists[1]
    dSums[1] = dists[0] + dists[2]
    dSums[2] = dists[1] + dists[2]
    mD = max(dSums)
    if len(fs) == 1:
        if dSums[0] == mD:
            if dists[0] > dists[1]:
                ireorder = (0, 1, 2)
                dreorder = (0, 1, 2)
            else:
                ireorder = (0, 2, 1)
                dreorder = (1, 0, 2)
        elif dSums[1] == mD:
            if dists[0] > dists[2]:
                ireorder = (1, 0, 2)
                dreorder = (0, 2, 1)
            else:
                ireorder = (1, 2, 0)
                dreorder = (2, 0, 1)
        elif dists[1] > dists[2]:
            ireorder = (2, 0, 1)
            dreorder = (1, 2, 0)
        else:
            ireorder = (2, 1, 0)
            dreorder = (2, 1, 0)
    elif featIndices[0] == featIndices[1]:
        if dists[1] > dists[2]:
            ireorder = (0, 1, 2)
            dreorder = (0, 1, 2)
        else:
            ireorder = (1, 0, 2)
            dreorder = (0, 2, 1)
    elif featIndices[0] == featIndices[2]:
        if dists[0] > dists[2]:
            ireorder = (0, 1, 2)
            dreorder = (0, 1, 2)
        else:
            ireorder = (2, 1, 0)
            dreorder = (2, 1, 0)
    elif dists[0] > dists[1]:
        ireorder = (0, 1, 2)
        dreorder = (0, 1, 2)
    else:
        ireorder = (0, 2, 1)
        dreorder = (1, 0, 2)
    dists = [dists[x] for x in dreorder]
    featIndices = [featIndices[x] for x in ireorder]
    return (featIndices, dists)