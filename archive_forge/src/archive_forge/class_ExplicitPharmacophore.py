import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
class ExplicitPharmacophore:
    """ this is a pharmacophore with explicit point locations and radii
  """

    def __init__(self, feats=None, radii=None):
        if feats and radii:
            self._initializeFeats(feats, radii)

    def _initializeFeats(self, feats, radii):
        if len(feats) != len(radii):
            raise ValueError('len(feats)!=len(radii)')
        self._feats = []
        self._radii = []
        for feat, rad in zip(feats, radii):
            if isinstance(feat, ChemicalFeatures.MolChemicalFeature):
                pos = feat.GetPos()
                newFeat = ChemicalFeatures.FreeChemicalFeature(feat.GetFamily(), feat.GetType(), Geometry.Point3D(pos[0], pos[1], pos[2]))
            else:
                newFeat = feat
            self._feats.append(newFeat)
            self._radii.append(rad)

    def getFeatures(self):
        return self._feats

    def getRadii(self):
        return self._radii

    def getFeature(self, i):
        return self._feats[i]

    def getRadius(self, i):
        return self._radii[i]

    def setRadius(self, i, rad):
        self._radii[i] = rad

    def initFromString(self, text):
        lines = text.split('\\n')
        self.initFromLines(lines)

    def initFromFile(self, inF):
        self.initFromLines(inF.readlines())

    def initFromLines(self, lines):
        import re
        spaces = re.compile('[ \t]+')
        feats = []
        rads = []
        for lineNum, line in enumerate(lines):
            txt = line.split('#')[0].strip()
            if txt:
                splitL = spaces.split(txt)
                if len(splitL) < 5:
                    logger.error(f'Input line {lineNum} only contains {len(splitL)} fields, 5 are required. Read failed.')
                    return
                fName = splitL[0]
                try:
                    xP = float(splitL[1])
                    yP = float(splitL[2])
                    zP = float(splitL[3])
                    rad = float(splitL[4])
                except ValueError:
                    logger.error(f'Error parsing a number of line {lineNum}. Read failed.')
                    return
                feats.append(ChemicalFeatures.FreeChemicalFeature(fName, fName, Geometry.Point3D(xP, yP, zP)))
                rads.append(rad)
        self._initializeFeats(feats, rads)

    def __str__(self):
        res = ''
        for feat, rad in zip(self._feats, self._radii):
            res += '% 12s ' % feat.GetFamily()
            p = feat.GetPos()
            res += '   % 8.4f % 8.4f % 8.4f    ' % (p.x, p.y, p.z)
            res += '% 5.2f' % rad
            res += '\n'
        return res