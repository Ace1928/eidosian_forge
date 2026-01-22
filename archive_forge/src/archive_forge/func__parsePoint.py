import re
from rdkit import Geometry
from rdkit.Chem.FeatMaps import FeatMapPoint, FeatMaps
def _parsePoint(self, txt):
    txt = txt.strip()
    startP = 0
    endP = len(txt)
    if txt[0] == '(':
        startP += 1
    if txt[-1] == ')':
        endP -= 1
    txt = txt[startP:endP]
    splitL = txt.split(',')
    if len(splitL) != 3:
        raise ValueError('Bad location string')
    return Geometry.Point3D(float(splitL[0]), float(splitL[1]), float(splitL[2]))