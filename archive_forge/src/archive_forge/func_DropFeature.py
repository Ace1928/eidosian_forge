import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
def DropFeature(self, i):
    del self._feats[i]