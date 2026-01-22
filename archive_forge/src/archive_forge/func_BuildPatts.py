import sys
from rdkit import Chem
def BuildPatts(rawV=None):
    """ Internal Use Only

  """
    global esPatterns, _rawD
    if rawV is None:
        rawV = _rawD
    esPatterns = [None] * len(rawV)
    for i, (name, sma) in enumerate(rawV):
        patt = Chem.MolFromSmarts(sma)
        if patt is None:
            sys.stderr.write('WARNING: problems with pattern %s (name: %s), skipped.\n' % (sma, name))
        else:
            esPatterns[i] = (name, patt)