import math
import numpy
from rdkit import Chem, Geometry

  Get the direction vectors for Acceptor of type 1

  This is a acceptor with one heavy atom neighbor. There are two possibilities we will
  consider here
  1. The bond to the heavy atom is a single bond e.g. CO
     In this case we don't know the exact direction and we just use the inversion of this bond direction
     and mark this direction as a 'cone'
  2. The bond to the heavy atom is a double bond e.g. C=O
     In this case the we have two possible direction except in some special cases e.g. SO2
     where again we will use bond direction
     
  ARGUMENTS:
    featAtoms - list of atoms that are part of the feature
    scale - length of the direction vector
  