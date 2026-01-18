import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def test_pathintegers(self):
    self.assertEqual(test1(), [{0: []}, {0: [1160, 2270], 1: [2260, 30692], 2: [1145, 33761]}, {0: [752, 1150, 1155, 3826, 38221, 43791], 1: [1145, 1150, 1596, 4670, 32641, 38211], 2: [1145, 1155, 5785, 32646, 43786, 65173]}])