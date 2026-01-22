from rdkit import RDLogger as logging
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.ChemUtils.AlignDepict import AlignDepict
def _exploder(mol, depth, sidechains, core, chainIndices, autoNames=True, templateName='', resetCounter=True, do3D=False, useTethers=False):
    global nDumped
    if resetCounter:
        nDumped = 0
    ourChains = sidechains[depth]
    patt = Chem.MolFromSmiles(f'[{depth + 1}*]')
    for i, (chainIdx, chain) in enumerate(ourChains):
        tchain = chainIndices[:]
        tchain.append((i, chainIdx))
        rs = Chem.ReplaceSubstructs(mol, patt, chain, replaceAll=True)
        if rs:
            r = rs[0]
            if depth < len(sidechains) - 1:
                for entry in _exploder(r, depth + 1, sidechains, core, tchain, autoNames=autoNames, templateName=templateName, resetCounter=0, do3D=do3D, useTethers=useTethers):
                    yield entry
            else:
                try:
                    Chem.SanitizeMol(r)
                except ValueError:
                    import traceback
                    traceback.print_exc()
                    continue
                if not do3D:
                    if r.HasSubstructMatch(core):
                        try:
                            AlignDepict(r, core)
                        except Exception:
                            import traceback
                            traceback.print_exc()
                            print(Chem.MolToSmiles(r), file=sys.stderr)
                    else:
                        print('>>> no match', file=sys.stderr)
                        AllChem.Compute2DCoords(r)
                else:
                    r = Chem.AddHs(r)
                    AllChem.ConstrainedEmbed(r, core, useTethers)
                Chem.Kekulize(r)
                if autoNames:
                    tName = 'TemplateEnum: Mol_%d' % (nDumped + 1)
                else:
                    tName = templateName
                    for bbI, bb in enumerate(tchain):
                        bbMol = sidechains[bbI][bb[0]][1]
                        if bbMol.HasProp('_Name'):
                            bbNm = bbMol.GetProp('_Name')
                        else:
                            bbNm = str(bb[1])
                        tName += '_' + bbNm
                r.SetProp('_Name', tName)
                r.SetProp('seq_num', str(nDumped + 1))
                r.SetProp('reagent_indices', '_'.join([str(x[1]) for x in tchain]))
                for bbI, bb in enumerate(tchain):
                    bbMol = sidechains[bbI][bb[0]][1]
                    if bbMol.HasProp('_Name'):
                        bbNm = bbMol.GetProp('_Name')
                    else:
                        bbNm = str(bb[1])
                    r.SetProp('building_block_%d' % (bbI + 1), bbNm)
                    r.SetIntProp('_idx_building_block_%d' % (bbI + 1), bb[1])
                    for propN in bbMol.GetPropNames():
                        r.SetProp('building_block_%d_%s' % (bbI + 1, propN), bbMol.GetProp(propN))
                nDumped += 1
                if not nDumped % 100:
                    logger.info('Done %d molecules' % nDumped)
                yield r