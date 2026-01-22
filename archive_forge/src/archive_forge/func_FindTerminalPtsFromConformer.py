import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def FindTerminalPtsFromConformer(conf, winRad, nbrCount):
    mol = conf.GetOwningMol()
    nAts = conf.GetNumAtoms()
    nbrLists = [[] for _ in range(nAts)]
    for i in range(nAts):
        if mol.GetAtomWithIdx(i).GetAtomicNum() <= 1:
            continue
        pi = conf.GetAtomPosition(i)
        nbrLists[i].append((i, pi))
        for j in range(i + 1, nAts):
            if mol.GetAtomWithIdx(j).GetAtomicNum() <= 1:
                continue
            pj = conf.GetAtomPosition(j)
            dist = pi.Distance(conf.GetAtomPosition(j))
            if dist < winRad:
                nbrLists[i].append((j, pj))
                nbrLists[j].append((i, pi))
    termPts = []
    while 1:
        for i in range(nAts):
            if not nbrLists[i]:
                continue
            pos = Geometry.Point3D(0, 0, 0)
            totWt = 0.0
            if len(nbrLists[i]) < nbrCount:
                nbrList = nbrLists[i]
                for j in range(0, len(nbrList)):
                    nbrJ, posJ = nbrList[j]
                    weight = 1.0 * len(nbrLists[i]) / len(nbrLists[nbrJ])
                    pos += posJ * weight
                    totWt += weight
                pos /= totWt
                termPts.append(SubshapeObjects.SkeletonPoint(location=pos))
        if not len(termPts):
            nbrCount += 1
        else:
            break
    return termPts