import sys
from rdkit import Chem
from rdkit.Chem import Randomize
def TestSupplier(suppl, stopAfter=-1, reportInterval=100, reportTo=sys.stderr, nameProp='_Name'):
    nDone = 0
    nFailed = 0
    while 1:
        try:
            mol = suppl.next()
        except StopIteration:
            break
        except Exception:
            import traceback
            traceback.print_exc()
            nFailed += 1
            reportTo.flush()
            print('Failure at mol %d' % nDone, file=reportTo)
        else:
            if mol:
                ok = TestMolecule(mol)
            else:
                ok = -3
            if ok < 0:
                nFailed += 1
                reportTo.flush()
                if ok == -3:
                    print('Canonicalization', end='', file=reportTo)
                print('Failure at mol %d' % nDone, end='', file=reportTo)
                if mol:
                    print(mol.GetProp(nameProp), end='', file=reportTo)
                print('', file=reportTo)
        nDone += 1
        if nDone == stopAfter:
            break
        if not nDone % reportInterval:
            print('Done %d molecules, %d failures' % (nDone, nFailed))
    return (nDone, nFailed)