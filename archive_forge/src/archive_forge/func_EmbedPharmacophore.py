import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def EmbedPharmacophore(mol, atomMatch, pcophore, randomSeed=-1, count=10, smoothFirst=True, silent=False, bounds=None, excludedVolumes=None, targetNumber=-1, useDirs=False):
    """ Generates one or more embeddings for a molecule that satisfy a pharmacophore

  atomMatch is a sequence of sequences containing atom indices
  for each of the pharmacophore's features.

    - count: is the maximum number of attempts to make a generating an embedding
    - smoothFirst: toggles triangle smoothing of the molecular bounds matix
    - bounds: if provided, should be the molecular bounds matrix. If this isn't
       provided, the matrix will be generated.
    - targetNumber: if this number is positive, it provides a maximum number
       of embeddings to generate (i.e. we'll have count attempts to generate
       targetNumber embeddings).

  returns: a 3 tuple:
    1) the molecular bounds matrix adjusted for the pharmacophore
    2) a list of embeddings (molecules with a single conformer)
    3) the number of failed attempts at embedding

    >>> from rdkit import Geometry
    >>> from rdkit.Chem.Pharm3D import Pharmacophore
    >>> m = Chem.MolFromSmiles('OCCN')
    >>> feats = [
    ...   ChemicalFeatures.FreeChemicalFeature('HBondAcceptor', 'HAcceptor1',
    ...                                        Geometry.Point3D(0.0, 0.0, 0.0)),
    ...   ChemicalFeatures.FreeChemicalFeature('HBondDonor', 'HDonor1',
    ...                                        Geometry.Point3D(2.65, 0.0, 0.0)),
    ...   ]
    >>> pcophore=Pharmacophore.Pharmacophore(feats)
    >>> pcophore.setLowerBound(0,1, 2.5)
    >>> pcophore.setUpperBound(0,1, 3.5)
    >>> atomMatch = ((0, ), (3, ))

    >>> bm,embeds,nFail = EmbedPharmacophore(m, atomMatch, pcophore, randomSeed=23, silent=1)
    >>> len(embeds)
    10
    >>> nFail
    0

    Set up a case that can't succeed:

    >>> pcophore = Pharmacophore.Pharmacophore(feats)
    >>> pcophore.setLowerBound(0,1, 2.0)
    >>> pcophore.setUpperBound(0,1, 2.1)
    >>> atomMatch = ((0, ), (3, ))

    >>> bm, embeds, nFail = EmbedPharmacophore(m, atomMatch, pcophore, randomSeed=23, silent=1)
    >>> len(embeds)
    0
    >>> nFail
    10

  """
    global _times
    if not hasattr(mol, '_chiralCenters'):
        mol._chiralCenters = Chem.FindMolChiralCenters(mol)
    if bounds is None:
        bounds = MolDG.GetMoleculeBoundsMatrix(mol)
    if smoothFirst:
        DG.DoTriangleSmoothing(bounds)
    bm = UpdatePharmacophoreBounds(bounds.copy(), atomMatch, pcophore, useDirs=useDirs, mol=mol)
    if excludedVolumes:
        bm = AddExcludedVolumes(bm, excludedVolumes, smoothIt=False)
    if not DG.DoTriangleSmoothing(bm):
        raise ValueError('could not smooth bounds matrix')
    if targetNumber <= 0:
        targetNumber = count
    nFailed = 0
    res = []
    for i in range(count):
        m2 = Chem.Mol(mol)
        t1 = time.perf_counter()
        try:
            if randomSeed <= 0:
                seed = i * 10 + 1
            else:
                seed = i * 10 + randomSeed
            EmbedMol(m2, bm.copy(), atomMatch, randomSeed=seed, excludedVolumes=excludedVolumes)
        except ValueError:
            if not silent:
                logger.info('Embed failed')
            nFailed += 1
        else:
            t2 = time.perf_counter()
            _times['embed'] = _times.get('embed', 0) + t2 - t1
            keepIt = True
            for idx, stereo in mol._chiralCenters:
                if stereo in ('R', 'S'):
                    vol = ComputeChiralVolume(m2, idx)
                    if stereo == 'R' and vol >= 0 or (stereo == 'S' and vol <= 0):
                        keepIt = False
                        break
            if keepIt:
                res.append(m2)
            else:
                logger.debug('Removed embedding due to chiral constraints.')
            if len(res) == targetNumber:
                break
    return (bm, res, nFailed)