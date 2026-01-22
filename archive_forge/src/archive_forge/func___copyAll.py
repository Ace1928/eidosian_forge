import copy
from rdkit.Chem.FeatMaps import FeatMaps
def __copyAll(res, fm1, fm2):
    """ no user-serviceable parts inside """
    for feat in fm1.GetFeatures():
        res.AddFeatPoint(copy.deepcopy(feat))
    for feat in fm2.GetFeatures():
        res.AddFeatPoint(copy.deepcopy(feat))