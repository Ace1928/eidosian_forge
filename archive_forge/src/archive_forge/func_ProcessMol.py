import re
import rdkit.RDLogger as logging
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def ProcessMol(mol, typeConversions, globalProps, nDone, nameProp='_Name', nameCol='compound_id', redraw=False, keepHs=False, skipProps=False, addComputedProps=False, skipSmiles=False, uniqNames=None, namesSeen=None):
    if not mol:
        raise ValueError('no molecule')
    if keepHs:
        Chem.SanitizeMol(mol)
    try:
        nm = mol.GetProp(nameProp)
    except KeyError:
        nm = None
    if not nm:
        nm = f'Mol_{nDone}'
    if uniqNames and nm in namesSeen:
        logger.error(f'duplicate compound id ({nm}) encountered. second instance skipped.')
        return None
    namesSeen.add(nm)
    row = [nm]
    pD = {}
    if not skipProps:
        if addComputedProps:
            nHD = Lipinski.NumHDonors(mol)
            mol.SetProp('DonorCount', str(nHD))
            nHA = Lipinski.NumHAcceptors(mol)
            mol.SetProp('AcceptorCount', str(nHA))
            nRot = Lipinski.NumRotatableBonds(mol)
            mol.SetProp('RotatableBondCount', str(nRot))
            MW = Descriptors.MolWt(mol)
            mol.SetProp('AMW', str(MW))
            logp = Crippen.MolLogP(mol)
            mol.SetProp('MolLogP', str(logp))
        pns = list(mol.GetPropNames())
        for pn in pns:
            if pn.lower() == nameCol.lower():
                continue
            pv = mol.GetProp(pn).strip()
            if pv.find('>') < 0 and pv.find('<') < 0:
                colTyp = globalProps.get(pn, 2)
                while colTyp > 0:
                    try:
                        _ = typeConversions[colTyp][1](pv)
                    except Exception:
                        colTyp -= 1
                    else:
                        break
                globalProps[pn] = colTyp
                pD[pn] = typeConversions[colTyp][1](pv)
            else:
                pD[pn] = pv
    if redraw:
        AllChem.Compute2DCoords(m)
    if not skipSmiles:
        row.append(Chem.MolToSmiles(mol))
    row.append(DbModule.binaryHolder(mol.ToBinary()))
    row.append(pD)
    return row