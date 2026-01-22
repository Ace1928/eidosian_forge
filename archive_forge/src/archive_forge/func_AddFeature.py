import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
def AddFeature(self, feat, weight=None):
    if self.params and (not feat.GetFamily() in self.params):
        raise ValueError('feature family %s not found in params' % feat.GetFamily())
    newFeat = FeatMapPoint()
    newFeat.initFromFeat(feat)
    newFeat.weight = weight
    self.AddFeatPoint(newFeat)