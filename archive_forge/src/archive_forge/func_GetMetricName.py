import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def GetMetricName(self):
    metricDict = {DataStructs.DiceSimilarity: 'Dice', DataStructs.TanimotoSimilarity: 'Tanimoto', DataStructs.CosineSimilarity: 'Cosine'}
    metric = metricDict.get(self.metric, self.metric)
    if metric:
        return metric
    return 'Unknown'