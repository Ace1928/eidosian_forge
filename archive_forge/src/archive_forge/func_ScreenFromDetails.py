import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import DbFpSupplier, FingerprintMols
from rdkit.DataStructs.TopNContainer import TopNContainer
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def ScreenFromDetails(details, mol=None):
    """ Returns a list of results

  """
    if not mol:
        if not details.probeMol:
            smi = details.probeSmiles
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:
                import traceback
                FingerprintMols.error(f'Error: problems generating molecule for smiles: {smi}\n')
                traceback.print_exc()
                return None
        else:
            mol = details.probeMol
    if not mol:
        return
    if details.outFileName:
        try:
            outF = open(details.outFileName, 'w+')
        except IOError:
            FingerprintMols.error(f'Error: could not open output file {details.outFileName} for writing\n')
            return None
    else:
        outF = None
    if not hasattr(details, 'useDbSimilarity') or not details.useDbSimilarity:
        res = ScreenFingerprints(details, data=GetFingerprints(details), mol=mol)
    else:
        res = ScreenInDb(details, mol)
    if outF:
        for pt in res:
            outF.write(','.join([str(x) for x in pt]))
            outF.write('\n')
    return res