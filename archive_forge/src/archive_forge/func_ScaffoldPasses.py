import itertools
def ScaffoldPasses(combo, bins=None):
    """ checks the scaffold passed in to see if all
    contributing triangles can satisfy the triangle inequality

    the scaffold itself (encoded in combo) is a list of binned distances

    """
    tris = GetTriangles(nDistPointDict[len(combo)])
    for tri in tris:
        ds = [bins[combo[x]] for x in tri]
        if not BinsTriangleInequality(ds[0], ds[1], ds[2]):
            return False
    return True