import copy
from rdkit.Chem.FeatMaps import FeatMaps
def CombineFeatMaps(fm1, fm2, mergeMetric=MergeMetric.NoMerge, mergeTol=1.5, dirMergeMode=DirMergeMode.NoMerge):
    """
     the parameters will be taken from fm1
  """
    res = FeatMaps.FeatMap(params=fm1.params)
    __copyAll(res, fm1, fm2)
    if mergeMetric != MergeMetric.NoMerge:
        MergeFeatPoints(res, mergeMetric=mergeMetric, mergeTol=mergeTol)
    return res