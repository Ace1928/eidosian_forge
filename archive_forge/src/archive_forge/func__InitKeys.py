from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
def _InitKeys(keyList, keyDict):
    """ *Internal Use Only*

   generates SMARTS patterns for the keys, run once

  """
    assert len(keyList) == len(keyDict.keys()), 'length mismatch'
    for key in keyDict.keys():
        patt, count = keyDict[key]
        if patt != '?':
            sma = Chem.MolFromSmarts(patt)
            if not sma:
                print('SMARTS parser error for key #%d: %s' % (key, patt))
            else:
                keyList[key - 1] = (sma, count)