import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory
def ShowDetails(catalog, gains, nToDo=-1, outF=sys.stdout, idCol=0, gainCol=1, outDelim=','):
    """
     gains should be a sequence of sequences.  The idCol entry of each
     sub-sequence should be a catalog ID.  _ProcessGainsData()_ provides
     suitable input.

    """
    if nToDo < 0:
        nToDo = len(gains)
    for i in range(nToDo):
        id_ = int(gains[i][idCol])
        gain = float(gains[i][gainCol])
        descr = catalog.GetFragDescription(id_)
        if descr:
            outF.write('%s\n' % outDelim.join((str(id_), descr, str(gain))))