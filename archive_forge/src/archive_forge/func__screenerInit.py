import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def _screenerInit(self):
    self.metric = DataStructs.TanimotoSimilarity
    self.doScreen = ''
    self.topN = 10
    self.screenThresh = 0.75
    self.doThreshold = 0
    self.smilesTableName = ''
    self.probeSmiles = ''
    self.probeMol = None
    self.noPickle = 0