import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
 Wrapper to calculate the matrix of TFD values for the
      conformers of a molecule.

      Arguments:
      - mol:      the molecule of interest
      - useWeights: flag for using torsion weights in the TFD calculation
      - maxDev:   maximal deviation used for normalization
                  'equal': all torsions are normalized using 180.0 (default)
                  'spec':  each torsion is normalized using its specific
                           maximal deviation as given in the paper
      - symmRadius: radius used for calculating the atom invariants
                    (default: 2)
      - ignoreColinearBonds: if True (default), single bonds adjacent to
                             triple bonds are ignored
                             if False, alternative not-covalently bound
                             atoms are used to define the torsion

      Return: matrix of TFD values
      Note that the returned matrix is symmetrical, i.e. it is the
      lower half of the matrix, e.g. for 5 conformers:
      matrix = [ a,
                 b, c,
                 d, e, f,
                 g, h, i, j]
  