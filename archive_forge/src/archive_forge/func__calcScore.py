import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _calcScore(reactantFP, productFP, bitInfoProd=None, output=False):
    if output:
        print('--- _calcScore ---')
    score = 0
    dFP = productFP - reactantFP
    numRBits = float(utils.getNumPositiveCounts(reactantFP))
    if output > 2:
        print('num RBits: ', numRBits)
    numPBits = float(utils.getNumPositiveCounts(productFP))
    if output > 2:
        print('num PBits: ', numPBits)
    numUnmappedPBits = float(utils.getNumPositiveCounts(dFP))
    if output > 2:
        print('num UnmappedPBits: ', numUnmappedPBits)
    numUnmappedRBits = float(utils.getNumNegativeCounts(dFP))
    if output > 2:
        print('num UnmappedRBits: ', numUnmappedRBits)
    numUnmappedPAtoms = -1
    bitsUnmappedPAtoms = -1
    if bitInfoProd is not None:
        numUnmappedPAtoms, bitsUnmappedPAtoms = utils.getNumPositiveBitCountsOfRadius0(dFP, bitInfoProd)
        if output > 2:
            print('num UnmappedPAtoms: ', numUnmappedPAtoms)
    ratioMappedPBits = 1 - numUnmappedPBits / numPBits
    ratioUnmappedRBits = numUnmappedRBits / numRBits
    score = max(ratioMappedPBits - ratioUnmappedRBits * ratioUnmappedRBits, 0)
    if output > 1:
        print('score: ', score, '(', ratioMappedPBits, ',', ratioUnmappedRBits * ratioUnmappedRBits, ',', ratioUnmappedRBits, ')')
    return [score, numUnmappedPBits, numUnmappedPAtoms, bitsUnmappedPAtoms]