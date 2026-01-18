import csv
import itertools
import logging
import math
import re
import sys
from collections import defaultdict, namedtuple
from typing import Generator, List
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, molzip
from rdkit.Chem import rdRGroupDecomposition as rgd
def test_rgroups():
    smiles = '[*:2]c1ccccc1[*:1]'
    rg = RGroup(smiles=smiles, rgroup='Core', count=1, coefficient=0.1)
    mol = Chem.MolFromSmiles(smiles)
    assert rg.mw == Descriptors.MolWt(mol)
    assert rg.hvyct == Descriptors.HeavyAtomCount(mol)
    assert rg.dummies == (1, 2)
    smiles = '[*:2]cccccc[*:1]'
    rg = RGroup(smiles=smiles, rgroup='Core', count=1, coefficient=0.1)
    assert rg.mw == 72.06599999999999
    assert rg.hvyct == 6
    assert rg.dummies == (1, 2)
    smiles = 'Nope'
    rg = RGroup(smiles=smiles, rgroup='Core', count=1, coefficient=0.1)
    assert rg.mw == 0
    assert rg.hvyct == 0
    assert rg.dummies == tuple()