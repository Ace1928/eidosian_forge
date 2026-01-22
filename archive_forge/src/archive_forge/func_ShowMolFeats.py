import math
from rdkit import RDLogger as logging
from rdkit import Geometry
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils
import os
import sys
from optparse import OptionParser
from rdkit import RDConfig
def ShowMolFeats(mol, factory, viewer, radius=0.5, confId=-1, showOnly=True, name='', transparency=0.0, colors=None, excludeTypes=[], useFeatDirs=True, featLabel=None, dirLabel=None, includeArrowheads=True, writeFeats=False, showMol=True, featMapFile=False):
    global _globalSphereCGO
    if not name:
        if mol.HasProp('_Name'):
            name = mol.GetProp('_Name')
        else:
            name = 'molecule'
    if not colors:
        colors = _featColors
    if showMol:
        viewer.ShowMol(mol, name=name, showOnly=showOnly, confId=confId)
    molFeats = factory.GetFeaturesForMol(mol)
    if not featLabel:
        featLabel = f'{name}-feats'
        viewer.server.resetCGO(featLabel)
    if not dirLabel:
        dirLabel = featLabel + '-dirs'
        viewer.server.resetCGO(dirLabel)
    for feat in molFeats:
        family = feat.GetFamily()
        if family in excludeTypes:
            continue
        pos = feat.GetPos(confId)
        color = colors.get(family, (0.5, 0.5, 0.5))
        if transparency:
            _globalSphereCGO.extend([ALPHA, 1 - transparency])
        else:
            _globalSphereCGO.extend([ALPHA, 1])
        _globalSphereCGO.extend([COLOR, color[0], color[1], color[2], SPHERE, pos.x, pos.y, pos.z, radius])
        if writeFeats:
            aidText = ' '.join([str(x + 1) for x in feat.GetAtomIds()])
            print(f'{family}\t{pos.x:.3f}\t{pos.y:.3f}\t{pos.z:.3f}\t1.0\t# {aidText}')
        if featMapFile:
            print(f'  family={family} pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) weight=1.0', end='', file=featMapFile)
        if useFeatDirs:
            ps = []
            if family == 'Aromatic':
                ps, _ = FeatDirUtils.GetAromaticFeatVects(mol.GetConformer(confId), feat.GetAtomIds(), pos, scale=1.0)
            elif family == 'Donor':
                aids = feat.GetAtomIds()
                if len(aids) == 1:
                    FeatVectsDictMethod = {1: FeatDirUtils.GetDonor1FeatVects, 2: FeatDirUtils.GetDonor2FeatVects, 3: FeatDirUtils.GetDonor3FeatVects}
                    featAtom = mol.GetAtomWithIdx(aids[0])
                    numHvyNbrs = len([1 for x in featAtom.GetNeighbors() if x.GetAtomicNum() > 1])
                    ps, _ = FeatVectsDictMethod[numHvyNbrs](mol.GetConformer(confId), aids, scale=1.0)
            elif family == 'Acceptor':
                aids = feat.GetAtomIds()
                if len(aids) == 1:
                    FeatVectsDictMethod = {1: FeatDirUtils.GetDonor1FeatVects, 2: FeatDirUtils.GetDonor2FeatVects, 3: FeatDirUtils.GetDonor3FeatVects}
                    featAtom = mol.GetAtomWithIdx(aids[0])
                    numHvyNbrs = len([x for x in featAtom.GetNeighbors() if x.GetAtomicNum() > 1])
                    ps, _ = FeatVectsDictMethod[numHvyNbrs](mol.GetConformer(confId), aids, scale=1.0)
            for tail, head in ps:
                ShowArrow(viewer, tail, head, radius, color, dirLabel, transparency=transparency, includeArrowhead=includeArrowheads)
                if featMapFile:
                    vect = head - tail
                    print(f'dir=({vect.x:.3f}, {vect.y:.3f}, {vect.z:.3f})', end='', file=featMapFile)
        if featMapFile:
            aidText = ' '.join([str(x + 1) for x in feat.GetAtomIds()])
            print(f'# {aidText}', file=featMapFile)