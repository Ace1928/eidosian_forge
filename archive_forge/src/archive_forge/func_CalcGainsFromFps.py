import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory
def CalcGainsFromFps(suppl, fps, topN=-1, actName='', acts=None, nActs=2, reportFreq=10, biasList=None):
    """ calculates info gains from a set of fingerprints

      *DOC*

    """
    nBits = len(fps[0])
    if topN < 0:
        topN = nBits
    if not actName and (not acts):
        actName = suppl[0].GetPropNames()[-1]
    if hasattr(suppl, '__len__'):
        nMols = len(suppl)
    else:
        nMols = -1
    if biasList:
        ranker = InfoTheory.InfoBitRanker(nBits, nActs, InfoTheory.InfoType.BIASENTROPY)
        ranker.SetBiasList(biasList)
    else:
        ranker = InfoTheory.InfoBitRanker(nBits, nActs, InfoTheory.InfoType.ENTROPY)
    for i, mol in enumerate(suppl):
        if not acts:
            try:
                act = int(mol.GetProp(actName))
            except KeyError:
                message('ERROR: Molecule has no property: %s\n' % actName)
                message('\tAvailable properties are: %s\n' % str(mol.GetPropNames()))
                raise KeyError(actName)
        else:
            act = acts[i]
        if i and (not i % reportFreq):
            if nMols > 0:
                message('Done %d of %d.\n' % (i, nMols))
            else:
                message('Done %d.\n' % i)
        fp = fps[i]
        ranker.AccumulateVotes(fp, act)
    gains = ranker.GetTopN(topN)
    return gains