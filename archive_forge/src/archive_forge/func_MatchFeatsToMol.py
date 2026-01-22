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
def MatchFeatsToMol(mol, featFactory, features):
    """ generates a list of all possible mappings of each feature to a molecule

  Returns a 2-tuple:
    1) a boolean indicating whether or not all features were found
    2) a list, numFeatures long, of sequences of features


    >>> import os.path
    >>> from rdkit import RDConfig, Geometry
    >>> fdefFile = os.path.join(RDConfig.RDCodeDir, 'Chem/Pharm3D/test_data/BaseFeatures.fdef')
    >>> featFactory = ChemicalFeatures.BuildFeatureFactory(fdefFile)
    >>> activeFeats = [
    ...  ChemicalFeatures.FreeChemicalFeature('Acceptor', Geometry.Point3D(0.0, 0.0, 0.0)),
    ...  ChemicalFeatures.FreeChemicalFeature('Donor', Geometry.Point3D(0.0, 0.0, 0.0))]
    >>> m = Chem.MolFromSmiles('FCCN')
    >>> match, mList = MatchFeatsToMol(m, featFactory, activeFeats)
    >>> match
    True

    Two feature types:

    >>> len(mList)
    2

    The first feature type, Acceptor, has two matches:

    >>> len(mList[0])
    2
    >>> mList[0][0].GetAtomIds()
    (0,)
    >>> mList[0][1].GetAtomIds()
    (3,)

    The first feature type, Donor, has a single match:

    >>> len(mList[1])
    1
    >>> mList[1][0].GetAtomIds()
    (3,)

  """
    molFeats = _getFeatDict(mol, featFactory, features)
    res = []
    for feat in features:
        matches = molFeats.get(feat.GetFamily(), [])
        if len(matches) == 0:
            return (False, None)
        res.append(matches)
    return (True, res)