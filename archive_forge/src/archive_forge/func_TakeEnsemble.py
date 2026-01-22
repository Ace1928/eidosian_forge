import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def TakeEnsemble(vect, ensembleIds, isDataVect=False):
    """

    >>> v = [10,20,30,40,50]
    >>> TakeEnsemble(v,(1,2,3))
    [20, 30, 40]
    >>> v = ['foo',10,20,30,40,50,1]
    >>> TakeEnsemble(v,(1,2,3),isDataVect=True)
    ['foo', 20, 30, 40, 1]

    """
    if isDataVect:
        ensembleIds = [x + 1 for x in ensembleIds]
        vect = [vect[0]] + [vect[x] for x in ensembleIds] + [vect[-1]]
    else:
        vect = [vect[x] for x in ensembleIds]
    return vect