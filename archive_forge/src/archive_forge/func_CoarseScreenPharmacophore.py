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
def CoarseScreenPharmacophore(atomMatch, bounds, pcophore, verbose=False):
    """
  >>> from rdkit import Geometry
  >>> from rdkit.Chem.Pharm3D import Pharmacophore
  >>> feats = [
  ...   ChemicalFeatures.FreeChemicalFeature('HBondAcceptor', 'HAcceptor1',
  ...                                        Geometry.Point3D(0.0, 0.0, 0.0)),
  ...   ChemicalFeatures.FreeChemicalFeature('HBondDonor', 'HDonor1',
  ...                                        Geometry.Point3D(2.65, 0.0, 0.0)),
  ...   ChemicalFeatures.FreeChemicalFeature('Aromatic', 'Aromatic1',
  ...                                        Geometry.Point3D(5.12, 0.908, 0.0)),
  ...   ]
  >>> pcophore = Pharmacophore.Pharmacophore(feats)
  >>> pcophore.setLowerBound(0, 1, 1.1)
  >>> pcophore.setUpperBound(0, 1, 1.9)
  >>> pcophore.setLowerBound(0, 2, 2.1)
  >>> pcophore.setUpperBound(0, 2, 2.9)
  >>> pcophore.setLowerBound(1, 2, 2.1)
  >>> pcophore.setUpperBound(1, 2, 3.9)

  >>> bounds = numpy.array([[0, 2, 3],[1, 0, 4],[2, 3, 0]], dtype=numpy.float64)
  >>> CoarseScreenPharmacophore(((0, ),(1, )),bounds, pcophore)
  True

  >>> CoarseScreenPharmacophore(((0, ),(2, )),bounds, pcophore)
  False

  >>> CoarseScreenPharmacophore(((1, ),(2, )),bounds, pcophore)
  False

  >>> CoarseScreenPharmacophore(((0, ),(1, ),(2, )),bounds, pcophore)
  True

  >>> CoarseScreenPharmacophore(((1, ),(0, ),(2, )),bounds, pcophore)
  False

  >>> CoarseScreenPharmacophore(((2, ),(1, ),(0, )),bounds, pcophore)
  False

  # we ignore the point locations here and just use their definitions:

  >>> feats = [
  ...   ChemicalFeatures.FreeChemicalFeature('HBondAcceptor', 'HAcceptor1',
  ...                                        Geometry.Point3D(0.0, 0.0, 0.0)),
  ...   ChemicalFeatures.FreeChemicalFeature('HBondDonor', 'HDonor1',
  ...                                        Geometry.Point3D(2.65, 0.0, 0.0)),
  ...   ChemicalFeatures.FreeChemicalFeature('Aromatic', 'Aromatic1',
  ...                                        Geometry.Point3D(5.12, 0.908, 0.0)),
  ...   ChemicalFeatures.FreeChemicalFeature('HBondDonor', 'HDonor1',
  ...                                        Geometry.Point3D(2.65, 0.0, 0.0)),
  ...                ]
  >>> pcophore=Pharmacophore.Pharmacophore(feats)
  >>> pcophore.setLowerBound(0,1, 2.1)
  >>> pcophore.setUpperBound(0,1, 2.9)
  >>> pcophore.setLowerBound(0,2, 2.1)
  >>> pcophore.setUpperBound(0,2, 2.9)
  >>> pcophore.setLowerBound(0,3, 2.1)
  >>> pcophore.setUpperBound(0,3, 2.9)
  >>> pcophore.setLowerBound(1,2, 1.1)
  >>> pcophore.setUpperBound(1,2, 1.9)
  >>> pcophore.setLowerBound(1,3, 1.1)
  >>> pcophore.setUpperBound(1,3, 1.9)
  >>> pcophore.setLowerBound(2,3, 1.1)
  >>> pcophore.setUpperBound(2,3, 1.9)
  >>> bounds = numpy.array([[0, 3, 3, 3], 
  ...                       [2, 0, 2, 2], 
  ...                       [2, 1, 0, 2], 
  ...                       [2, 1, 1, 0]], 
  ...                      dtype=numpy.float64)

  >>> CoarseScreenPharmacophore(((0, ), (1, ), (2, ), (3, )), bounds, pcophore)
  True

  >>> CoarseScreenPharmacophore(((0, ), (1, ), (3, ), (2, )), bounds, pcophore)
  True

  >>> CoarseScreenPharmacophore(((1, ), (0, ), (3, ), (2, )), bounds, pcophore)
  False

  """
    atomMatchSize = len(atomMatch)
    for k in range(atomMatchSize):
        if len(atomMatch[k]) == 1:
            for l in range(k + 1, atomMatchSize):
                if len(atomMatch[l]) == 1:
                    if atomMatch[l][0] < atomMatch[k][0]:
                        idx0, idx1 = (atomMatch[l][0], atomMatch[k][0])
                    else:
                        idx0, idx1 = (atomMatch[k][0], atomMatch[l][0])
                    if bounds[idx1, idx0] >= pcophore.getUpperBound(k, l) or bounds[idx0, idx1] <= pcophore.getLowerBound(k, l):
                        if verbose:
                            print(f'\t  ({idx1},{idx0}) [{k},{l}] fail')
                            print(f'\t    {bounds[idx1, idx0]},{pcophore.getUpperBound(k, l)} - {bounds[idx0, idx1]},{pcophore.getLowerBound(k, l)}')
                        return False
    return True