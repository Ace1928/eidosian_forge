import copy
from rdkit.Chem.FeatMaps import FeatMaps
class MergeMetric(object):
    NoMerge = 0
    Distance = 1
    Overlap = 2

    @classmethod
    def valid(cls, mergeMetric):
        """ Check that mergeMetric is valid """
        if mergeMetric not in (cls.NoMerge, cls.Distance, cls.Overlap):
            raise ValueError('unrecognized mergeMetric')