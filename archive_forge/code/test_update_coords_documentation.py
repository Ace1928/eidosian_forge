import numpy as np
from pytest import mark
from ase import Atoms

    Check that the coordinates registered with the KIM API are updated
    appropriately when the atomic positions are updated.  This can go awry
    if the 'coords' attribute of the relevant NeighborList subclass is
    reassigned to a new memory location -- a problem which was indeed
    occurring at one point (see https://gitlab.com/ase/ase/merge_requests/1442)!
    