import pickle
from rdkit import DataStructs
from rdkit.VLib.Node import VLibNode
def _BuildFp(self, data):
    data = list(data)
    pkl = bytes(data[self.fpCol], encoding='Latin1')
    del data[self.fpCol]
    self._numProcessed += 1
    try:
        if self._usePickles:
            newFp = pickle.loads(pkl, encoding='bytes')
        else:
            newFp = DataStructs.ExplicitBitVect(pkl)
    except Exception:
        import traceback
        traceback.print_exc()
        newFp = None
    if newFp:
        newFp._fieldsFromDb = data
    return newFp