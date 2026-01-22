import re
import rdkit.RDLogger as logging
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def ConvertRows(rows, globalProps, defaultVal, skipSmiles):
    for i, row in enumerate(rows):
        newRow = [row[0], row[1]]
        pD = row[-1]
        newRow.extend((pD.get(pn, defaultVal) for pn in globalProps))
        newRow.append(row[2])
        if not skipSmiles:
            newRow.append(row[3])
        rows[i] = newRow