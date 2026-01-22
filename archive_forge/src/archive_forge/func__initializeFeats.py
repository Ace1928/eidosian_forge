import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
def _initializeFeats(self, feats, weights):
    self._feats = []
    if feats:
        if len(feats) != len(weights):
            raise ValueError('feats and weights lists must be the same length')
        for feat, weight in zip(feats, weights):
            self.AddFeature(feat, weight)