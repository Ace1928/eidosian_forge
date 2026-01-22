import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def _clusterInit(self):
    self.clusterAlgo = Murtagh.WARDS
    self.actTableName = ''
    self.actName = ''