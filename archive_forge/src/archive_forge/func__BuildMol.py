import sys
from rdkit import Chem
from rdkit.Chem.Suppliers.MolSupplier import MolSupplier
def _BuildMol(self, data):
    data = list(data)
    molD = data[self.molCol]
    del data[self.molCol]
    self._numProcessed += 1
    try:
        if self.molFmt == 'SMI':
            newM = Chem.MolFromSmiles(str(molD))
            if not newM:
                warning('Problems processing mol %d, smiles: %s\n' % (self._numProcessed, molD))
        elif self.molFmt == 'PKL':
            newM = Chem.Mol(str(molD))
    except Exception:
        import traceback
        traceback.print_exc()
        newM = None
    else:
        if newM and self.transformFunc:
            try:
                newM = self.transformFunc(newM, data)
            except Exception:
                import traceback
                traceback.print_exc()
                newM = None
        if newM:
            newM._fieldsFromDb = data
            nFields = len(data)
            for i in range(nFields):
                newM.SetProp(self._colNames[i], str(data[i]))
            if self.nameCol >= 0:
                newM.SetProp('_Name', str(data[self.nameCol]))
    return newM